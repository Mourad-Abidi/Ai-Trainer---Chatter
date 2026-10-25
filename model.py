# model.py
"""Character-level GRU language model.

Design notes
------------
* Multi-layer GRU + LayerNorm on the classifier input. LayerNorm stabilises
  training at higher learning rates, which is where this kind of model
  actually converges fastest.

* Orthogonal init on the GRU *recurrent* weights and Xavier on the *input*
  weights. This is the standard recipe (Zaremba et al. 2014) and prevents
  the exploding/vanishing gradient see-saw that otherwise dominates the
  first few thousand steps.

* Dropout is applied three places: embedding output, between GRU layers
  (only if num_layers > 1), and before the classifier.

* Optional weight tying (Press & Wolf, 2017): share the input embedding and
  the output projection. Saves parameters AND regularises. Requires
  embedding_dim == hidden_size. Best for tiny models.

* Optional ``init_output_bias(log_prior)``: initialise the output-layer bias
  to ``log(p(char))`` for the corpus unigram distribution. Step 0 already
  predicts letters at the right frequency instead of uniform noise.

Streaming-generation note
-------------------------
``generate_tokens`` is a *generator*. Decorating a generator function with
``@torch.no_grad()`` does NOT disable autograd during iteration: the
decorator opens the context, the function call returns a generator object
without running its body, and the context closes before iteration starts.
The no_grad context is therefore entered *inside* the generator body here.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import torch
from torch import nn

# ─────────────────────────────────────────────────────────────────────────
#  Project paths — the fix for "checkpoint not found: ai_trainer---chatter.pt"
# ─────────────────────────────────────────────────────────────────────────
# Root cause of that error: the checkpoint used to be a *bare filename*,
# resolved against whatever directory the shell happened to be in. Train from
# one folder, chat from another (or chat before training finished) and the
# file "does not exist". Now:
#
#   * SAVING always anchors relative paths to PROJECT_DIR (the folder that
#     holds this file), never to the shell's current directory.
#   * LOADING looks in PROJECT_DIR *and* the current directory (so older
#     checkpoints keep working) and picks the newest match.
#   * If the default checkpoint is missing, the newest ``*.pt`` next to the
#     code is used as a fallback instead of crashing.
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_NAME = "ai_trainer---chatter.pt"
DEFAULT_DATA_DIRNAME = "data_to_train_on"


def resolve_checkpoint_path(path=None) -> Path:
    """Absolute path a checkpoint should be WRITTEN to.

    ``None``/empty → the default name. Relative paths are anchored to
    ``PROJECT_DIR`` rather than the current working directory, so training
    and chatting always agree on where the file lives.
    """
    p = Path(str(path)).expanduser() if path else Path(DEFAULT_CHECKPOINT_NAME)
    return p if p.is_absolute() else (PROJECT_DIR / p)


def find_checkpoint(path=None, *, allow_fallback: bool = True) -> Path | None:
    """Locate an existing checkpoint to READ, or return ``None``.

    Search order (the newest existing file wins):
      1. the path as given, if absolute
      2. ``PROJECT_DIR / path``
      3. ``CWD / path``  (back-compat with checkpoints saved the old way)
      4. only when the *default* name was requested and nothing was found:
         the newest ``*.pt`` in ``PROJECT_DIR`` or the CWD.

    A custom, explicitly-named checkpoint that does not exist is never
    silently replaced by a different model.
    """
    name = Path(str(path)).expanduser() if path else Path(DEFAULT_CHECKPOINT_NAME)
    if name.is_absolute():
        candidates = [name]
    else:
        candidates = [PROJECT_DIR / name, Path.cwd() / name]
    existing = [c for c in candidates if c.is_file()]
    if existing:
        return max(existing, key=lambda c: c.stat().st_mtime).resolve()

    if allow_fallback and name.name == DEFAULT_CHECKPOINT_NAME:
        pool: list[Path] = []
        for folder in {PROJECT_DIR, Path.cwd()}:
            try:
                pool.extend(f for f in folder.glob("*.pt") if f.is_file())
            except OSError:
                pass
        if pool:
            return max(pool, key=lambda c: c.stat().st_mtime).resolve()
    return None


def atomic_torch_save(obj, path) -> None:
    """``torch.save`` that can never leave a half-written checkpoint.

    Writes to ``<name>.tmp`` then renames over the target (atomic on every
    OS Python supports). Killing the process mid-save leaves the previous
    good checkpoint intact instead of a corrupt file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def load_torch_file(path, device="cpu"):
    """``torch.load`` that prefers the safe ``weights_only`` mode.

    Checkpoints written by train.py contain only tensors, numbers, strings,
    lists and dicts, so the safe loader handles them. If a PyTorch version
    rejects something, fall back to the classic loader (the file is the
    user's own).
    """
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:                       # very old PyTorch: no weights_only=
        return torch.load(path, map_location=device)
    except Exception:                       # unpickling restriction -> retry
        return torch.load(path, map_location=device, weights_only=False)


# Layers whose first dimension is the vocabulary size.
_VOCAB_KEYS = ("embedding.weight", "output.weight", "output.bias")


@torch.no_grad()
def load_state_expanding_vocab(model: nn.Module, state: dict, old_vocab: int) -> int:
    """Load ``state`` into ``model``; the model may have a LARGER vocabulary.

    This is what lets a second training session use text that contains
    characters the first session never saw: the new characters are appended
    to the vocabulary (old ids never move), every old weight is kept, and
    only the brand-new rows start fresh.

    Returns the number of vocabulary rows that were added (0 = exact match).
    Raises ``RuntimeError`` when the shapes cannot be reconciled (for example
    a different hidden size) so the caller can fall back to a fresh model.
    """
    own = model.state_dict()
    added = 0
    for key, src in state.items():
        if key not in own:
            raise RuntimeError(f"checkpoint has unexpected weight '{key}'")
        dst = own[key]
        src = src.to(dst.device, dst.dtype)
        if src.shape == dst.shape:
            dst.copy_(src)
        elif (key in _VOCAB_KEYS and src.shape[0] == old_vocab
              and dst.shape[0] > old_vocab and src.shape[1:] == dst.shape[1:]):
            dst[:old_vocab].copy_(src)
            if key == "output.bias":        # new chars: start as 'rare'
                dst[old_vocab:] = src.min()
            added = dst.shape[0] - old_vocab
        else:
            raise RuntimeError(
                f"weight '{key}' has shape {tuple(src.shape)} in the "
                f"checkpoint but {tuple(dst.shape)} in the new model")
    missing = [k for k in own if k not in state]
    if missing:
        raise RuntimeError(f"checkpoint is missing weight '{missing[0]}'")
    return added


def describe_checkpoint(ckpt: dict) -> str:
    """One human-readable line about a loaded checkpoint dict."""
    steps = int(ckpt.get("total_steps", ckpt.get("step", 0)) or 0)
    sessions = int(ckpt.get("sessions", 1) or 1)
    best = ckpt.get("best_validation_loss")
    bits = f"{steps:,} steps over {sessions} session{'s' if sessions != 1 else ''}"
    if isinstance(best, (int, float)) and best == best and best != float("inf"):
        bits += f", best validation loss {best:.4f}"
    return bits


class EMA:
    """Exponential moving average of a model's parameters.

    Averaging the weights over the last few thousand steps removes much of
    the SGD noise: the averaged model almost always has a lower validation
    loss than the raw weights, for free (no extra gradient steps). Use
    :meth:`average_weights` around evaluation, sampling and checkpointing.

    The decay is warmed up (``(1+n)/(10+n)``) so the average is useful from
    the very first evaluations instead of lagging behind a random init.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}")
        self.decay = float(decay)
        self.updates = 0
        # model.parameters() yields tied weights only once, matching the
        # order used in update()/swap — so tying needs no special casing.
        self.shadow = [p.detach().clone() for p in model.parameters()]

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        for s, p in zip(self.shadow, model.parameters()):
            s.lerp_(p.detach(), 1.0 - d)

    @contextmanager
    def average_weights(self, model: nn.Module):
        """Temporarily load the averaged weights into ``model``."""
        params = list(model.parameters())
        backup = [p.detach().clone() for p in params]
        with torch.no_grad():
            for p, s in zip(params, self.shadow):
                p.copy_(s)
        try:
            yield model
        finally:
            with torch.no_grad():
                for p, b in zip(params, backup):
                    p.copy_(b)


class CharGRU(nn.Module):
    """Predict the next character in a text sequence."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        tie_weights: bool = False,
    ) -> None:
        super().__init__()
        if vocab_size < 2:
            raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")
        if tie_weights and embedding_dim != hidden_size:
            raise ValueError(
                f"tie_weights requires embedding_dim == hidden_size "
                f"(got {embedding_dim} and {hidden_size})"
            )

        # Architecture metadata — useful for model cards / tooltips without
        # round-tripping through the checkpoint dict.
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.tie_weights = bool(tie_weights)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embed_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.out_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, vocab_size, bias=True)
        if tie_weights:
            self.output.weight = self.embedding.weight
        self._init_weights()

    # ── introspection ─────────────────────────────────────────────────
    @property
    def num_parameters(self) -> int:
        """Total learnable parameters (accounts for weight tying)."""
        return sum(p.numel() for p in self.parameters())

    # ── initialisation ────────────────────────────────────────────────
    def _init_weights(self) -> None:
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.normal_(self.embedding.weight, std=0.1)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    @torch.no_grad()
    def init_output_bias(self, log_prior: torch.Tensor) -> None:
        """Seed the output layer with a log unigram prior.

        Parameters
        ----------
        log_prior : Tensor of shape ``(vocab_size,)``
            ``log(p(char))`` measured on the training corpus. Replaces the
            zero init, which would otherwise predict uniform noise at
            step 0 and waste hundreds of warmup steps.
        """
        if self.output.bias is None:
            raise RuntimeError("output layer has no bias to initialise")
        if log_prior.shape != self.output.bias.shape:
            raise ValueError(
                f"log_prior shape {tuple(log_prior.shape)} does not match "
                f"vocab size {self.output.bias.numel()}"
            )
        self.output.bias.copy_(log_prior.to(self.output.bias.device))

    # ── forward ───────────────────────────────────────────────────────
    def forward(
        self,
        token_ids: torch.Tensor,
        hidden: torch.Tensor | None = None,
        return_hidden: bool = False,
    ):
        """Return one set of next-character logits per input character."""
        x = self.embed_dropout(self.embedding(token_ids))
        states, hidden = self.gru(x, hidden)
        logits = self.output(self.out_dropout(self.norm(states)))
        return (logits, hidden) if return_hidden else logits

    # ── sampling ──────────────────────────────────────────────────────
    @staticmethod
    def _pick(
        logits: torch.Tensor,
        temperature: float,
        top_k: int | None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        recent: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Sample one next token.

        ``repetition_penalty`` > 1 discourages characters seen in ``recent``
        (the CTRL-style divide/multiply rule); it fights the looping that
        small character models are prone to.
        ``temperature <= 0`` → greedy argmax.
        ``top_k`` of ``None`` or ``0`` → no filtering.
        ``top_p`` (nucleus) keeps the smallest set of tokens whose
        probability mass reaches ``top_p``; ``None``/``>=1`` → off.
        """
        if repetition_penalty and repetition_penalty != 1.0 and recent \
                and logits.size(0) == 1:
            logits = logits.clone()
            ids = torch.tensor(sorted(set(int(i) for i in recent)),
                               device=logits.device, dtype=torch.long)
            picked = logits[0, ids]
            logits[0, ids] = torch.where(
                picked > 0, picked / repetition_penalty,
                picked * repetition_penalty)

        if temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        logits = logits / float(temperature)

        # Keep *exactly* the top-k logits (index-based masking; the
        # ``< kth`` threshold trick leaks ties past k).
        if top_k is not None and 0 < top_k < logits.size(-1):
            idx = torch.topk(logits, top_k, dim=-1).indices
            mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                -1, idx, True)
            logits = logits.masked_fill(~mask, float("-inf"))

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(
                logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            # Drop a token once the mass *before* it already reaches top_p;
            # the most likely token is therefore always kept.
            drop = (sorted_probs.cumsum(dim=-1) - sorted_probs) > top_p
            sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(
                -1, sorted_idx, sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    # ── generation: batched ───────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        block_size: int | None = None,  # kept for backwards compatibility
        stop_token_id: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        repeat_window: int = 32,
    ) -> torch.Tensor:
        """Continue a prompt (batched, non-streaming).

        For streaming / interactive UIs, use :meth:`generate_tokens`.
        """
        if max_new_tokens <= 0:
            return token_ids

        was_training = self.training
        self.eval()
        try:
            logits, hidden = self(token_ids, return_hidden=True)
            out = token_ids
            next_logits = logits[:, -1, :]
            recent = token_ids[0, -repeat_window:].tolist() \
                if token_ids.size(0) == 1 else None
            for _ in range(max_new_tokens):
                nxt = self._pick(next_logits, temperature, top_k, top_p,
                                 repetition_penalty, recent)
                if recent is not None:
                    recent.append(int(nxt[0, 0]))
                    del recent[:-repeat_window]
                out = torch.cat((out, nxt), dim=1)
                if stop_token_id is not None and int(nxt[0, 0]) == stop_token_id:
                    break
                logits, hidden = self(nxt, hidden, return_hidden=True)
                next_logits = logits[:, -1, :]
            return out
        finally:
            if was_training:
                self.train()

    # ── generation: streaming ─────────────────────────────────────────
    def generate_tokens(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        stop_token_id: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        should_stop=None,  # optional callable() -> bool checked each step
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        repeat_window: int = 32,
    ) -> Iterator[int]:
        """Yield one token id at a time. Used by the streaming chat UI.

        The hidden state is carried forward, so each new token costs one
        GRU step instead of re-reading the prompt.

        ``should_stop`` is polled between tokens; when it returns True the
        generator returns cleanly. This is the cooperative cancellation
        mechanism used by the GUI's Stop button.
        """
        if max_new_tokens <= 0:
            return

        was_training = self.training
        self.eval()
        try:
            # NOTE: no_grad must be entered *inside* the generator body —
            # see the module docstring for why.
            with torch.no_grad():
                logits, hidden = self(prompt_ids, return_hidden=True)
                next_logits = logits[:, -1, :]
                recent = prompt_ids[0, -repeat_window:].tolist()
                for _ in range(max_new_tokens):
                    if should_stop is not None and should_stop():
                        return
                    nxt = self._pick(next_logits, temperature, top_k, top_p,
                                     repetition_penalty, recent)
                    cid = int(nxt[0, 0])
                    recent.append(cid)
                    del recent[:-repeat_window]
                    yield cid
                    if stop_token_id is not None and cid == stop_token_id:
                        return
                    logits, hidden = self(nxt, hidden, return_hidden=True)
                    next_logits = logits[:, -1, :]
        finally:
            if was_training:
                self.train()

    # ── checkpoint I/O ────────────────────────────────────────────────
    @classmethod
    def from_checkpoint(cls, path, device=None):
        """Load a CharGRU + its metadata from a checkpoint written by train.py."""
        found = find_checkpoint(path)
        if found is None:
            raise FileNotFoundError(
                f"Model checkpoint not found: {resolve_checkpoint_path(path)}. "
                "Run train.py first (python train.py --cli --data "
                f"{DEFAULT_DATA_DIRNAME})."
            )
        path = found
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = load_torch_file(path, device)

        required = ("characters", "embedding_dim", "hidden_size")
        missing = [k for k in required if k not in ckpt]
        if missing:
            raise RuntimeError(
                f"Checkpoint is missing required field(s): {missing}. "
                "Retrain with the current train.py."
            )

        model = cls(
            vocab_size=len(ckpt["characters"]),
            embedding_dim=ckpt["embedding_dim"],
            hidden_size=ckpt["hidden_size"],
            num_layers=ckpt.get("num_layers", 2),
            dropout=0.0,  # dropout is off at inference
            tie_weights=ckpt.get("tie_weights", False),
        ).to(device)

        try:
            model.load_state_dict(ckpt["model_state_dict"])
        except RuntimeError as exc:
            raise RuntimeError(
                "This checkpoint does not match the current model "
                "architecture (made by an older train.py?). Retrain with "
                f"train.py.\nUnderlying error: {str(exc).splitlines()[0]}"
            ) from exc

        model.eval()
        return model, ckpt, device