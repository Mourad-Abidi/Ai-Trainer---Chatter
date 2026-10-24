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

from pathlib import Path
from typing import Iterator

import torch
from torch import nn


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
        logits: torch.Tensor, temperature: float, top_k: int | None
    ) -> torch.Tensor:
        """Sample one next token.

        ``temperature <= 0`` → greedy argmax.
        ``top_k`` of ``None`` or ``0`` → no filtering.
        """
        if temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        logits = logits / float(temperature)

        # Keep *exactly* the top-k logits. The threshold trick
        # (``logits < kth``) accidentally keeps ties with the k-th value,
        # which can leak more than k candidates through when quantised
        # logits collide. Indices-based masking is exact.
        if top_k is not None and 0 < top_k < logits.size(-1):
            idx = torch.topk(logits, top_k, dim=-1).indices
            mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                -1, idx, True)
            logits = logits.masked_fill(~mask, float("-inf"))

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
            for _ in range(max_new_tokens):
                nxt = self._pick(next_logits, temperature, top_k)
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
                for _ in range(max_new_tokens):
                    if should_stop is not None and should_stop():
                        return
                    nxt = self._pick(next_logits, temperature, top_k)
                    cid = int(nxt[0, 0])
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
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {path}. Run train.py first."
            )
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            ckpt = torch.load(path, map_location=device, weights_only=True)
        except TypeError:  # older PyTorch without weights_only=
            ckpt = torch.load(path, map_location=device)

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
                "This checkpoint was made by the old model architecture — "
                "please retrain with train.py."
            ) from exc

        model.eval()
        return model, ckpt, device