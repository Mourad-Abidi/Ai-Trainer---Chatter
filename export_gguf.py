"""Export PyTorch trained checkpoints (.pt) to GGUF format for Ollama / LM Studio / Bionic / llama.cpp.

Usage:
    python export_gguf.py --checkpoint ai_trainer---chatter.pt --out ai_trainer_model.gguf
"""

from __future__ import annotations

import argparse
import io
import os
import struct
import sys
from pathlib import Path
from typing import Any, Sequence

try:
    import torch
except ImportError:
    torch = None


# ── Pure Python GGUF Writer (Zero external dependencies) ───────────────────
class PureGGUFWriter:
    """Pack metadata and tensors into a valid GGUF v3 file."""

    # GGUF Constants
    MAGIC = 0x46554747  # "GGUF"
    VERSION = 3

    # Value Types
    TYPE_UINT8 = 0
    TYPE_INT8 = 1
    TYPE_UINT16 = 2
    TYPE_INT16 = 3
    TYPE_UINT32 = 4
    TYPE_INT32 = 5
    TYPE_FLOAT32 = 6
    TYPE_BOOL = 7
    TYPE_STRING = 8
    TYPE_ARRAY = 9
    TYPE_UINT64 = 10
    TYPE_INT64 = 11
    TYPE_FLOAT64 = 12

    # GGML Tensor Types
    GGML_TYPE_F32 = 0
    GGML_TYPE_F16 = 1

    def __init__(self, path: Path | str, arch: str = "custom_gru"):
        self.path = Path(path)
        self.arch = arch
        self.kv_metadata: list[tuple[str, int, bytes]] = []
        self.tensors: list[dict[str, Any]] = []

    def _pack_str(self, val: str) -> bytes:
        encoded = val.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded

    def add_string(self, key: str, value: str):
        val_bytes = self._pack_str(value)
        self.kv_metadata.append((key, self.TYPE_STRING, val_bytes))

    def add_uint32(self, key: str, value: int):
        val_bytes = struct.pack("<I", int(value))
        self.kv_metadata.append((key, self.TYPE_UINT32, val_bytes))

    def add_int32(self, key: str, value: int):
        val_bytes = struct.pack("<i", int(value))
        self.kv_metadata.append((key, self.TYPE_INT32, val_bytes))

    def add_float32(self, key: str, value: float):
        val_bytes = struct.pack("<f", float(value))
        self.kv_metadata.append((key, self.TYPE_FLOAT32, val_bytes))

    def add_bool(self, key: str, value: bool):
        val_bytes = struct.pack("?", bool(value))
        self.kv_metadata.append((key, self.TYPE_BOOL, val_bytes))

    def add_string_array(self, key: str, values: Sequence[str]):
        arr_bytes = bytearray()
        arr_bytes.extend(struct.pack("<I", self.TYPE_STRING))
        arr_bytes.extend(struct.pack("<Q", len(values)))
        for v in values:
            arr_bytes.extend(self._pack_str(v))
        self.kv_metadata.append((key, self.TYPE_ARRAY, bytes(arr_bytes)))

    def add_float_array(self, key: str, values: Sequence[float]):
        arr_bytes = bytearray()
        arr_bytes.extend(struct.pack("<I", self.TYPE_FLOAT32))
        arr_bytes.extend(struct.pack("<Q", len(values)))
        for v in values:
            arr_bytes.extend(struct.pack("<f", float(v)))
        self.kv_metadata.append((key, self.TYPE_ARRAY, bytes(arr_bytes)))

    def add_tensor(self, name: str, tensor: Any):
        if torch is not None and isinstance(tensor, torch.Tensor):
            arr = tensor.detach().cpu().to(torch.float32).contiguous().numpy()
        elif hasattr(tensor, "astype"):
            arr = tensor.astype("float32")
        else:
            arr = tensor
        shape = list(arr.shape)
        # Reverse shape for GGUF dimension convention (GGML uses column-major / reverse shape)
        gguf_shape = list(reversed(shape))
        raw_data = arr.tobytes()
        self.tensors.append({
            "name": name,
            "shape": gguf_shape,
            "type": self.GGML_TYPE_F32,
            "data": raw_data,
        })

    def write(self):
        alignment = 32
        self.add_uint32("general.alignment", alignment)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            # 1. Header
            header = struct.pack(
                "<IIQQ",
                self.MAGIC,
                self.VERSION,
                len(self.tensors),
                len(self.kv_metadata)
            )
            f.write(header)

            # 2. KV Metadata
            for key, val_type, val_bytes in self.kv_metadata:
                f.write(self._pack_str(key))
                f.write(struct.pack("<I", val_type))
                f.write(val_bytes)

            # 3. Calculate tensor offsets & write tensor metadata
            offset_accumulator = 0
            for t in self.tensors:
                t["offset"] = offset_accumulator
                offset_accumulator += len(t["data"])
                # Align tensor data start
                padding = (alignment - (offset_accumulator % alignment)) % alignment
                offset_accumulator += padding

            for t in self.tensors:
                f.write(self._pack_str(t["name"]))
                n_dims = len(t["shape"])
                f.write(struct.pack("<I", n_dims))
                for dim in t["shape"]:
                    f.write(struct.pack("<Q", dim))
                f.write(struct.pack("<I", t["type"]))
                f.write(struct.pack("<Q", t["offset"]))

            # 4. Alignment padding before tensor binary block
            curr_pos = f.tell()
            pad = (alignment - (curr_pos % alignment)) % alignment
            if pad > 0:
                f.write(b"\x00" * pad)

            # 5. Tensor binary data
            tensor_start_pos = f.tell()
            for t in self.tensors:
                expected_pos = tensor_start_pos + t["offset"]
                curr = f.tell()
                if curr < expected_pos:
                    f.write(b"\x00" * (expected_pos - curr))
                f.write(t["data"])


# ── High-level Exporter ───────────────────────────────────────────────────
def export_checkpoint_to_gguf(checkpoint_path: Path | str,
                               output_path: Path | str) -> Path:
    """Load PyTorch checkpoint file and export it as a GGUF file."""
    ckpt_p = Path(checkpoint_path)
    out_p = Path(output_path)

    if not ckpt_p.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_p}")

    # Load checkpoint safely
    from model import load_torch_file
    ckpt = load_torch_file(ckpt_p, device="cpu")

    characters: list[str] = ckpt.get("characters", [])
    if not characters:
        # Fallback if characters list is in model weights or stoi
        stoi = ckpt.get("stoi", {})
        if stoi:
            characters = [k for k, _ in sorted(stoi.items(), key=lambda x: x[1])]
        else:
            characters = [chr(i) for i in range(256)]

    vocab_size = int(ckpt.get("vocab_size", len(characters)))
    embedding_dim = int(ckpt.get("embedding_dim", 64))
    hidden_size = int(ckpt.get("hidden_size", 512))
    num_layers = int(ckpt.get("num_layers", 2))
    tie_weights = bool(ckpt.get("tie_weights", False))
    arch = "custom_gru"

    # Use 'gguf' package if available, else PureGGUFWriter
    try:
        import gguf
        writer = gguf.GGUFWriter(str(out_p), arch)
        writer.add_string("general.architecture", arch)
        writer.add_string("general.name", "ai-trainer-chatter")
        writer.add_uint32("custom_gru.vocab_size", vocab_size)
        writer.add_uint32("custom_gru.embedding_dim", embedding_dim)
        writer.add_uint32("custom_gru.hidden_size", hidden_size)
        writer.add_uint32("custom_gru.num_layers", num_layers)
        writer.add_bool("custom_gru.tie_weights", tie_weights)
        writer.add_tokenizer_model("gpt2")
        writer.add_token_list(characters)
        writer.add_token_scores([0.0] * len(characters))

        # Add weights
        state_dict = ckpt.get("model", ckpt)
        if isinstance(state_dict, dict):
            for k, v in state_dict.items():
                if isinstance(v, torch.Tensor):
                    writer.add_tensor(k, v.detach().cpu().numpy())
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
    except ImportError:
        writer = PureGGUFWriter(out_p, arch=arch)
        writer.add_string("general.architecture", arch)
        writer.add_string("general.name", "ai-trainer-chatter")
        writer.add_uint32("custom_gru.vocab_size", vocab_size)
        writer.add_uint32("custom_gru.embedding_dim", embedding_dim)
        writer.add_uint32("custom_gru.hidden_size", hidden_size)
        writer.add_uint32("custom_gru.num_layers", num_layers)
        writer.add_bool("custom_gru.tie_weights", tie_weights)
        writer.add_string_array("tokenizer.ggml.tokens", characters)
        writer.add_float_array("tokenizer.ggml.scores", [0.0] * len(characters))

        state_dict = ckpt.get("model", ckpt)
        if isinstance(state_dict, dict):
            for k, v in state_dict.items():
                if isinstance(v, torch.Tensor):
                    writer.add_tensor(k, v)
        writer.write()

    return out_p


def main():
    parser = argparse.ArgumentParser(description="Export Ai-Trainer checkpoint to GGUF format.")
    parser.add_argument("--checkpoint", "-c", default="ai_trainer---chatter.pt",
                        help="Path to PyTorch checkpoint file (.pt)")
    parser.add_argument("--out", "-o", default="ai_trainer_model.gguf",
                        help="Output GGUF file path (.gguf)")
    args = parser.parse_args()

    try:
        out_path = export_checkpoint_to_gguf(args.checkpoint, args.out)
        print(f"Successfully exported model to GGUF: {out_path} ({out_path.stat().st_size:,} bytes)")
    except Exception as exc:
        print(f"Error exporting to GGUF: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
