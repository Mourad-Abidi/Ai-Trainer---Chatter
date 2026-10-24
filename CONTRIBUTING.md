# Contributing to AI Trainer & Chatter

Thanks for considering a contribution. This project is deliberately small and readable — the ideal place to send your first PyTorch PR, improve a GUI, or sharpen documentation. The goal of this guide is to make that process straightforward and to keep the codebase easy to reason about.

By participating, you agree to be respectful and constructive. Harassment and hostile behaviour are not tolerated.

---

## Table of Contents

- [Ways to Contribute](#-ways-to-contribute)
- [Setting Up a Development Environment](#️-setting-up-a-development-environment)
- [Project Architecture](#-project-architecture)
- [Manual Smoke Tests](#-manual-smoke-tests)
- [Coding Conventions](#-coding-conventions)
- [Recipe: Add a New Training CLI Flag](#-recipe-add-a-new-training-cli-flag)
- [Recipe: Add a New GUI Control](#-recipe-add-a-new-gui-control)
- [Recipe: Add a Chat Slash Command](#-recipe-add-a-chat-slash-command)
- [Recipe: Add a Dashboard Panel](#-recipe-add-a-dashboard-panel)
- [Recipe: Add a New Dataset Generator](#-recipe-add-a-new-dataset-generator)
- [Recipe: Add a New Model Architecture](#-recipe-add-a-new-model-architecture)
- [Adding Tests](#-adding-tests)
- [Gotchas Specific to This Codebase](#️-gotchas-specific-to-this-codebase)
- [Opening a Pull Request](#-opening-a-pull-request)
- [Reporting Bugs and Requesting Features](#-reporting-bugs-and-requesting-features)
- [Data Privacy and Responsible Corpora](#-data-privacy-and-responsible-corpora)

---

## ✨ Ways to Contribute

### Documentation

- Fix typos or stale examples in `README.md`.
- Expand the **Loss-Reduction Playbook** with a new experiment you ran.
- Add a short guide for a specific corpus (e.g., "training on a Python codebase").
- Add screenshots of the training launcher and chat window to the README.

### Bug fixes

- Any crash, traceback, or obviously wrong output is fair game.
- Look for unguarded `OSError`, `RuntimeError`, `KeyboardInterrupt`, or `tk.TclError`.
- Improve error messages that currently say "something went wrong" without telling the user what to do.

### Visualization

- Add new dashboard panels (see [the recipe](#-recipe-add-a-dashboard-panel)).
- Improve the styling of `plot_results.py` for narrow screens.
- Add per-run annotations (e.g., a marker where the best checkpoint was saved).

### Model & training

- Experiment with architectural alternatives (Transformer, LSTM, hybrid).
- Add `--resume` (continue from a checkpoint).
- Add a proper `train / val / test` split.
- Add gradient noise, SWA, or other stabilization techniques — as opt-in flags, not replacements.

### Chat & GUI

- Add keyboard shortcuts, transcript search, or a "copy last response" button.
- Make the GUI more accessible: focus order, screen-reader labels, larger hit targets.
- Improve the CLI REPL (e.g., history navigation, tab completion).

### Tooling

- Add `requirements.txt` / `pyproject.toml`.
- Set up GitHub Actions for lint + tests.
- Add a `pre-commit` hook (ruff or black + isort).

---

## 🛠️ Setting Up a Development Environment

```bash
git clone https://github.com/Mourad-Abidi/Ai-Trainer---Chatter.git
cd Ai-Trainer---Chatter
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
```

Install PyTorch using the [official instructions](https://pytorch.org/get-started/locally/) for your OS and accelerator, then:

```bash
pip install numpy matplotlib pandas
```

**Optional but recommended for development:**

```bash
pip install ruff pytest
```

Verify the environment:

```bash
python -c "import torch, numpy, matplotlib, pandas; print('ok')"
python -c "import tkinter; print('tkinter ok')" || echo "(tkinter missing — CLI only)"
```

> **Windows note:** Tkinter ships with the standard installer. On Fedora use `sudo dnf install python3-tkinter`; on Debian/Ubuntu use `sudo apt install python3-tk`.

---

## 📂 Project Architecture

Familiarise yourself with these before touching anything. The whole project is ~2500 lines and broken cleanly across five files.

| File | Responsibility |
|---|---|
| `model.py` | `CharGRU` architecture, `generate()`, `generate_tokens()`, `from_checkpoint()`. No I/O, no argparse, no side effects. |
| `train.py` | Data resolution (`expand_data_specs`, `load_corpus`), argument validation, training loop, GUI launcher, CLI parser. |
| `ask.py` | `ChatWindow` (GUI), `run_cli` (REPL), settings persistence, checkpoint loading orchestration. |
| `plot_results.py` | `load_log`, `draw`, `save_static`, `run_live` — pure functions on a CSV file. |
| `generate_data.py` | Synthetic arithmetic corpus generator. Standalone; no shared state. |

### Data flow

```
files ─► expand_data_specs ─► load_corpus ─► char vocab ─► encode_text
                                                              │
                                              ┌───────────────┴───────────────┐
                                              │                               │
                                          train_data                      val_data
                                              │                               │
                                              ▼                               │
                                        CharGRU  ◄──── AdamW, AMP, clipping   │
                                              │                               │
                                              │◄──────── eval_loss ───────────┘
                                              ▼
                                    tiny_text_model.pt + training_log.csv
                                                              │
                                              ┌───────────────┴───────────────┐
                                              ▼                               ▼
                                     plot_results.py --live               ask.py
```

### Contracts between files

Three contracts must stay coherent — breaking one of them silently breaks another:

1. **Checkpoint schema.** Written by `train.py`, read by `model.CharGRU.from_checkpoint`. Fields: `model_state_dict`, `characters`, `block_size`, `embedding_dim`, `hidden_size`, `num_layers`, `dropout`, `tie_weights`, `best_validation_loss`, `step`. Add fields freely; **never rename or remove them** without a migration path.
2. **CSV log schema.** `LOG_COLUMNS` in `train.py` must match `NUMERIC` in `plot_results.py` for every numeric column. `prepare_log` auto-rotates the CSV when the header changes, so adding a column is safe for existing users — as long as `plot_results.py` is updated in the same PR.
3. **Settings file.** `~/.tiny_text_ai_ask.json` uses the `DEFAULTS` dict in `ask.py`. Add keys, never remove; use `settings.get(key, DEFAULTS[key])` when reading.

---

## 🧪 Manual Smoke Tests

There are no automated tests yet (see [Adding Tests](#-adding-tests)). Until that changes, every PR should include a manual smoke test. These are the fastest ones:

```bash
# 1. Train end-to-end on the bundled corpus (about 1–2 min on CPU)
python train.py --cli --steps 500 --batch-size 32 --block-size 128 \
    --eval-interval 100 --no-live --no-auto-chat
# Expect: loss decreases, tiny_text_model.pt created, no traceback.

# 2. Chat with the checkpoint you just made
python ask.py --cli --no-stream
# Type one prompt. Expect: a response, no crash. Try /help, /stats, /model.

# 3. Plot the log you just produced
python plot_results.py
# Expect: training_dashboard.png, no warning about missing columns.

# 4. Chat GUI (if tkinter available)
python ask.py
# Type a prompt, press Enter, verify streaming, try Stop (Esc), Save (Ctrl+S).

# 5. Training GUI (if tkinter available)
python train.py --steps 100 --eval-interval 50
# Verify every field is pre-filled, help badges work, Start spawns a
# subprocess that hits `step 100`, Stop stops it.

# 6. Data generator
python generate_data.py --max-number 5 --output /tmp/math5.txt
# Expect: a small Q&A file, no crash.
```

Include which of these you ran in your PR description.

---

## 🧹 Coding Conventions

- **Python 3.11+.** Use `X | None` unions, `from __future__ import annotations`, `Path` over `os.path`, `dataclass` where it genuinely simplifies.
- **Type hints on public functions.** Internal helpers may skip them.
- **No new dependencies without justification.** `numpy` and `pandas` are already optional-ish (`_np` guard in `train.py`); follow that pattern if you must add one.
- **Keep the CLI first-class.** Any feature added to the GUI must also be reachable via `--cli` (or documented as GUI-only with a reason).
- **No silent failures.** Every `except` should either re-raise, log to stdout/stderr with context, or handle explicitly. Never `pass` on a real error.
- **Error messages are documentation.** Say what went wrong *and* what the user should try. Compare:

  ```python
  # bad
  raise SystemExit("error")
  # good
  raise SystemExit("error: --tie-weights requires --embedding-dim == --hidden-size")
  ```

- **Determinism is a feature.** Any new randomness must flow from `--seed` (or a caller-supplied RNG).
- **Preserve existing behaviour unless the PR is about changing it.** Small surprises add up.

---

## 🍳 Recipe: Add a New Training CLI Flag

Every training flag follows the same four-step pattern. Example: adding `--eval-max-batches`.

### 1. Declare it in `build_parser()`

In `train.py`, inside `build_parser()`, in the right section:

```python
p.add_argument("--eval-max-batches", type=int, default=0,
               help="Cap eval_batches at this many batches per eval. "
                    "0 = no cap.")
```

### 2. Validate it in `_validate_training_args()`

```python
if args.eval_max_batches < 0:
    raise SystemExit("error: --eval-max-batches must be >= 0")
```

### 3. Consume it in `run_training()`

```python
n_eval_batches = args.eval_batches
if args.eval_max_batches:
    n_eval_batches = min(n_eval_batches, args.eval_max_batches)
```

### 4. Expose it in the GUI

In `TrainingLauncher._build()`, add the field next to its siblings:

```python
field(form, 17, "Eval max batches", "eval_max_batches", 0, width=14)
```

In `_build_cmd()`, add the key to the list of forwarded flags:

```python
for key in [..., "eval_batches", "eval_max_batches", ...]:
```

And in `_validate()`, add it to `int_keys`.

Add a help entry to the `HELP` dict at the top of `train.py`:

```python
"eval_max_batches": "Cap on eval_batches per validation pass. 0 = no cap.",
```

### 5. Document it

Add a row to the README's [Hyperparameters Reference](./README.md#️-hyperparameters-reference) table.

---

## 🍳 Recipe: Add a New GUI Control

The GUI system uses `field()`, `pair()`, `_help_badge()`, and `Tooltip`. The pattern:

```python
# 1. Add to HELP at the top of the file
HELP["my_control"] = "What it does, in one sentence, aimed at a beginner."

# 2. Add a form row (use pair() for a two-column layout)
pair(form, N, "My control", "my_control", <default>,
     "Other control", "other_control", <default>)

# 3. Read it in _validate() and _build_cmd() as shown above.
```

Rules:

- **Every control has a `?` badge and a tooltip.** No exceptions — the whole point of the GUI is discoverability.
- **Buttons have `clickable=False` on their tooltip.** Otherwise the tooltip swallows the click. `_help_badge()` handles this automatically.
- **Don't introduce state that isn't persisted.** If you add a durable toggle to `ask.py`'s GUI, add it to `DEFAULTS` so it survives restarts.

---

## 🍳 Recipe: Add a Chat Slash Command

Every command in `ask.py`'s CLI REPL is a branch inside `_handle_command`. To add `/echo`:

```python
# inside _handle_command, before the final "unknown command" branch
if name == "echo":
    print(c.dim(f"  {val}"))
    return True
```

Then update `_handle_command`'s `/help` output so the command is discoverable:

```python
print("    /echo <text>       print text back (debugging)")
```

If the command changes sampling state (like `/temp`), also update the `/stats` output so the change is visible.

GUI equivalents (if applicable) go through `HELP` entries and buttons in `_build_header()` or the sampling sidebar.

---

## 🍳 Recipe: Add a Dashboard Panel

`plot_results.py` is driven by two parallel lists. Adding a panel is a three-file change:

### 1. Emit the metric from `train.py`

If the metric isn't already in the CSV:

```python
# in train.py
LOG_COLUMNS = [..., "my_metric"]
```

And emit it in the `writer.writerow([...])` call inside the training loop. Whatever expression you use, keep it cheap — this runs on every `--log-interval` steps.

### 2. Add the metric to `plot_results.py`

```python
NUMERIC = [..., "my_metric"]

PANELS = [
    # ...
    ("my_metric", "Human-readable title", "unit"),
]
```

The `PANELS` list is `(key, title, y-label)` triples. `draw()` iterates this list and calls `ax.plot(sub["step"], sub[key])` — the key must match a column in the CSV.

### 3. Verify

```bash
python plot_results.py --out /tmp/test_dashboard.png
# Open the PNG. There should be a new subplot between the existing panels.
```

### Notes

- `PANELS` is 8 entries today, laid out as 2×4 in `make_figure()`. Adding a 9th entry will silently overflow the grid — either resize `plt.subplots(2, 4)` to a compatible shape (`2, 5` or `3, 4`), or clamp to 8 by removing a panel first.
- `prepare_log` in `train.py` rotates `training_log.csv` when `LOG_COLUMNS` changes, so old CSVs won't crash `plot_results.py`. Document the rotation in your PR — users will see it.

---

## 🍳 Recipe: Add a New Dataset Generator

`generate_data.py` is a self-contained arithmetic generator. If you want to add a new generator (e.g., `generate_dialogues.py`):

### 1. Create a sibling file

```
generate_dialogues.py
```

Follow the pattern of `generate_data.py`:

- `argparse` with a `description=__doc__`.
- `generate_*()` function returning a `list[str]` of examples.
- A `main()` that writes to a `--output` file, **appending** if it exists.
- A clear example of the output format in the module docstring.

### 2. Preserve the Q&A format

The bundled chat client's `--qa` mode wraps prompts as `Q: {question}\nA:` and stops at the first newline. If your generator produces a different format, either:

- Adopt the `Q: …\nA: …` format (recommended — zero friction for users), or
- Document that the user must run `ask.py` in **non-Q&A mode**.

### 3. Add a short section to the README

A sentence in [Synthetic data](#-synthetic-data-generate_datapy) plus a copy-pasteable example. Screenshots are welcome but not required.

---

## 🍳 Recipe: Add a New Model Architecture

`CharGRU` is the only model. If you want to introduce a Transformer or LSTM:

### 1. Understand the interface contract

`train.py` and `ask.py` both depend on exactly three things:

- `__init__(vocab_size, embedding_dim, hidden_size, num_layers, dropout, tie_weights)`
- `forward(token_ids, hidden=None, return_hidden=False)` returning `logits` or `(logits, hidden)`
- `generate_tokens(prompt_ids, max_new_tokens, stop_token_id, temperature, top_k, should_stop)` yielding `int` ids

Plus the introspection property `num_parameters` and the classmethod `from_checkpoint`.

### 2. Options, in increasing scope

| Scope | Approach |
|---|---|
| Personal experiment | Copy `model.py` to `my_model.py`, adapt the class. Don't touch `train.py` yet. |
| Repo-wide addition | Add a `--arch {gru,lstm,transformer}` flag to `train.py` and pick the class at instantiation. Save the arch name in the checkpoint. Update `from_checkpoint` to dispatch on it. |
| Full generalization | Introduce a `ModelBase` protocol, refactor `from_checkpoint` to take the arch name and look up the class in a registry. Only worth it once there are 3+ architectures. |

### 3. The checkpoint schema must stay backwards-compatible

If `--arch` is missing from an old checkpoint, assume `"gru"`. This is not optional — the README tells users their old checkpoints will keep working.

### 4. Do not break `generate_tokens`

The generator must:

- Yield one token id per iteration.
- Return (not yield a sentinel) when `should_stop()` is True.
- Restore the original `training` flag in a `finally` block.

The GUI's `_poll` relies on this: the worker wraps it in try/except and *always* sends `("done", stats)` in a `finally` — the UI will hang if you break that.

---

## 🧪 Adding Tests

Tests don't exist yet. The highest-value additions are listed in the roadmap. If you're picking this up:

### Suggested layout

```
tests/
├── test_model.py             # forward pass shapes, checkpoint round-trip
├── test_train_data.py        # expand_data_specs, load_corpus, encode_text
├── test_plot_results.py      # load_log on synthetic CSVs, draw() smoke test
└── test_generate_data.py     # calculate(), generate_examples() correctness
```

### Suggested `pytest` targets

```python
# test_model.py
def test_char_gru_forward_shape():
    m = CharGRU(vocab_size=10, embedding_dim=8, hidden_size=16)
    x = torch.zeros(2, 5, dtype=torch.long)
    assert m(x).shape == (2, 5, 10)

def test_checkpoint_round_trip(tmp_path):
    m = CharGRU(10, 8, 16).eval()
    ckpt = {"model_state_dict": m.state_dict(), "characters": list("abcdefghij"),
            "embedding_dim": 8, "hidden_size": 16, "num_layers": 2,
            "dropout": 0.0, "tie_weights": False,
            "best_validation_loss": 1.0, "step": 1}
    torch.save(ckpt, tmp_path / "m.pt")
    m2, ckpt2, device = CharGRU.from_checkpoint(tmp_path / "m.pt")
    assert m2.num_parameters == m.num_parameters

# test_train_data.py
def test_expand_data_specs_dedupes(tmp_path):
    (tmp_path / "a.txt").write_text("hi")
    files = expand_data_specs([str(tmp_path / "a.txt"), str(tmp_path)], (".txt",))
    assert len(files) == 1

def test_load_corpus_counts_replacements(tmp_path):
    (tmp_path / "bad.txt").write_bytes(b"hi\xff")
    text, counts, replaced = load_corpus([tmp_path / "bad.txt"])
    assert "\ufffd" in text and replaced >= 1
```

### Rules for new tests

- No network access. No GPU required. No tkinter.
- Every test runs in under 5 seconds.
- Use `tmp_path` for anything that touches disk.
- Prefer checking *behaviour* (shapes, round-trips, error messages) over internal details.

---

## ⚠️ Gotchas Specific to This Codebase

Read these before touching the files they refer to. They are all real, current footguns.

### In `model.py`

- **`@torch.no_grad()` on a generator does not work.** The decorator opens the context, the function call returns a generator object without running the body, and the context closes before iteration starts. That's why `generate_tokens` enters `torch.no_grad()` *inside* the body. Do not "clean this up" by decorating.
- **`_pick` uses index-based top-k masking**, not a threshold. The threshold trick keeps ties at the k-th value and can leak more than `k` candidates through when logits collide. This was deliberate.
- **`forward(return_hidden=True)` is load-bearing** for generation. Both `generate` and `generate_tokens` depend on receiving `(logits, hidden)`.

### In `train.py`

- **`scaler.unscale_(optimizer)` must run before `clip_grad_norm_`.** Otherwise you're clipping scaled gradients. Anyone touching the AMP path must preserve this order.
- **`get_batch` uses `torch.randint(..., device=data.device)`** — but `data` must be `.to(device)` first. It is, at line ~"`train_data, val_data = train_data.to(device), val_data.to(device)`". Do not move the training loop before that line.
- **`prepare_log` rotates the CSV when `LOG_COLUMNS` changes.** Anyone adding a column must also update `plot_results.NUMERIC` in the same PR, or the dashboard will error on new logs.
- **The GUI subprocess must be launched with `--cli`.** Otherwise `train.py` re-opens the GUI recursively. This is commented in `_build_cmd`; do not remove it.
- **Temp manifests must be cleaned up on all exit paths.** `_start()` calls `_cleanup_temp_manifest()` in every branch — on failure to spawn, on `_poll_log` completion, and in `_on_close`. If you add an early return to `_start`, add cleanup there too.

### In `ask.py`

- **`_token_queue` must be a fresh queue per generation.** If you reuse the old one, stale `("done", …)` messages from the previous run will finish the new generation on the first poll tick. This is why `_start_generation` re-instantiates it.
- **The worker always sends `("done", stats)` in its `finally` block.** Even on exception, the error handler path drains the queue expecting that trailing `done`. If you add a new message type, update `_poll`'s loop and `_drain_queue` together.
- **`_load_worker` must capture `exc` as a string before scheduling the Tk callback.** Python deletes the exception binding at the end of the `except` block, so `lambda: ...exc...` would raise `NameError`. The current code does `err = str(exc)` then `lambda msg=err: ...`. Preserve that pattern.
- **The `lambda m=model, c=ckpt, d=device:` default-argument binding** is similarly essential — Tk callbacks run after `_load_worker` returns, and without the binding they'd see `None` for all three.

### In `plot_results.py`

- **The backend is set with `matplotlib.use("Agg", force=True)`** in `save_static`, but `run_live` requires a GUI backend. If you import `plot_results` from `train.py`'s post-training dashboard save, be aware that this is a deliberate split.
- **`load_log` returns `None`, never raises.** Callers test `if df is None:`. Preserve that.
- **`cache_frame_data=False`** in `FuncAnimation` is required to avoid a `UserWarning` and a memory leak across refreshes. Don't remove it.

### In `generate_data.py`

- **`--max-number` is clamped to 99** regardless of what the user passes, because the output size grows as `O(n²)`. If you raise the ceiling, print a size estimate first.

---

## 🚀 Opening a Pull Request

### 1. Fork and branch

```bash
git clone https://github.com/<your-username>/Ai-Trainer---Chatter.git
cd Ai-Trainer---Chatter
git remote add upstream https://github.com/Mourad-Abidi/Ai-Trainer---Chatter.git
git switch -c fix/<short-description>       # or feature/<name>, docs/<name>
```

Branch name prefixes we use:

| Prefix | Purpose |
|---|---|
| `fix/` | Bug fix |
| `feature/` | New functionality |
| `docs/` | Documentation only |
| `refactor/` | Code movement without behaviour change |
| `test/` | Test-only additions |

### 2. Commit

One logical change per commit. Commit messages in the imperative mood:

```
Add --eval-max-batches cap for large validation sets
Fix token queue reuse causing premature generation finish
Document the @torch.no_grad-on-generator pitfall in model.py
```

Avoid: `update`, `fix bug`, `wip`, `asdf`.

If your commit is substantial, add a short body explaining *why* — the "why" is often more useful than the "what".

### 3. Test before pushing

Run the relevant [smoke tests](#-manual-smoke-tests). If you added a CLI flag, run `python train.py --help` and verify the flag appears. If you added a dashboard panel, generate a fresh PNG and confirm the new subplot renders.

If `pytest` is set up by the time you read this, run it:

```bash
pytest -q
ruff check .  # optional
```

### 4. Push and open the PR

```bash
git push -u origin fix/<short-description>
```

Open the PR against `main`. Include:

- **What changed** — one paragraph.
- **Why** — the problem this solves.
- **How you tested it** — which smoke tests, on which OS/device.
- **Backwards compatibility** — anything that could break old checkpoints, old CSVs, or the settings file.
- **Screenshots** — for any UI change.
- **Known limitations** — anything you consciously left out.

PRs that skip the "how you tested" section will be asked to add it before merge. This is not gatekeeping — the maintainers can't reproduce your environment.

### 5. Review and iterate

- Push follow-up commits to the same branch; the PR updates automatically.
- Force-push is fine *before* the first review, but avoid it afterwards — reviewers lose track of what changed.
- Once approved and CI passes, squash-merge is used for small PRs, merge-commit for multi-commit feature PRs.

### PR size

Prefer PRs under ~400 lines of diff. If your change is bigger, split it:

1. **Refactor PR** — pure code movement, no behaviour change.
2. **Behaviour PR** — small, testable, references the refactor.

This is not strict, but large PRs take much longer to review and merge.

---

## 🐛 Reporting Bugs and Requesting Features

### Bug report

Open an issue with:

- **Environment** — Python version, PyTorch version, OS, CPU/GPU model.
- **What you ran** — the exact command, including flags.
- **Expected behaviour.**
- **Actual behaviour** — including the full traceback.
- **Minimal reproduction** — ideally a small corpus and a command that reproduces the issue in under a minute.

If the bug is a crash on a specific corpus, try `python train.py --cli --steps 1 --batch-size 2 --block-size 16 --no-live --no-auto-chat` — if it still crashes, the corpus is enough to attach (a hundred lines is usually sufficient).

### Feature request

Describe:

- **The problem you're solving** — not the specific solution, but the pain point.
- **What you've tried.**
- **Why this project is the right home for it.**

Features that align with the [README roadmap](./README.md#️-roadmap) are especially welcome.

### Questions and discussions

Open an issue with the `question` label. The maintainers prefer public issues over private emails because the answers help the next person.

---

## 🔐 Data Privacy and Responsible Corpora

This project trains on **your** text. That text ends up baked into a checkpoint — which is a **distribution of training patterns**, not a lossless copy, but it is still recoverable in fragments. The maintainer will not merge PRs that:

- Ship corpora containing personal conversations, private records, credentials, or confidential information.
- Ship corpora scraped from sites whose terms prohibit redistribution.
- Ship corpora that could be used to generate harmful or deceptive content.

If you want to contribute a **new dataset generator** (see the recipe above), ship the *generator*, not the output. The maintainer will not merge large generated corpora into the repository — attach them as a release artifact if needed.

---

## 🙏 Thanks

Every contribution — typo fix, dashboard panel, or a completely new architecture — makes the project better for the next person who lands here. Thank you for taking the time.

If you're not sure where to start, look for issues labelled `good first issue` or open a discussion and ask.
