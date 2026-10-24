# Tiny Math AI — the smallest useful training pipeline

This is a learning project, not a reliable calculator. It demonstrates the complete loop:

1. Generate example questions and answers.
2. Convert characters to integer token IDs.
3. Train a small neural network to predict the next character.
4. Give it an unseen expression and inspect its prediction.

The network is a character-level **GRU language model** implemented with PyTorch. It has an embedding, one GRU layer, and a linear output layer. Training uses next-character targets and cross-entropy loss.

## 1. Use your existing Python environment

On your Fedora computer, using the environment you already created:

```bash
cd ~/Dev/products
unzip ~/Downloads/tiny_math_ai.zip
cd tiny_math_ai

# Check that PyTorch works in your AI environment:
~/Dev/envs/ai/bin/python -c 'import torch; print(torch.__version__); print("CUDA:", torch.cuda.is_available())'
```

Adjust the `unzip` path if you saved the ZIP somewhere else. Alternatively, extract the ZIP using your file manager.

## 2. Generate a small first dataset

Start with **addition only** and operands from 0 through 99. This gives the model one pattern to learn before you make the task harder.

```bash
~/Dev/envs/ai/bin/python generate_data.py --count 50000 --max-number 99 --operators +
```

The generated file `math_data.txt` contains records like:

```text
Q: 17 + 8
A: 25

Q: 42 + 6
A: 48
```

You can inspect this text file directly. `generate_data.py` calculates the correct labels; the neural model does not have access to the calculation function during training or inference.

## 3. Train the model

```bash
~/Dev/envs/ai/bin/python train.py --steps 3000
```

You will see training loss and validation loss. The best validation checkpoint is saved to `tiny_math_model.pt`.

- **Training loss** measures next-character prediction on sampled parts of the training text.
- **Validation loss** measures it on examples held out from training.
- A decreasing loss shows better text prediction. It does **not** prove correct arithmetic on new numbers.

## 4. Ask questions

```bash
~/Dev/envs/ai/bin/python ask.py
```

Try `12 + 7`, `50 + 49`, or an expression not seen in the training data. The answer may be wrong, incomplete, or blank. That is a useful result to observe, not a software crash: this first model predicts likely text rather than executing arithmetic rules.

## 5. Expand after the first experiment

First measure how often it gets unseen examples right. Then generate data with more operators:

```bash
~/Dev/envs/ai/bin/python generate_data.py --count 100000 --max-number 99 --operators + - '*'
~/Dev/envs/ai/bin/python train.py --steps 5000
```

For integer division too:

```bash
~/Dev/envs/ai/bin/python generate_data.py --count 100000 --max-number 99 --operators + - '*' '//'
~/Dev/envs/ai/bin/python train.py --steps 5000
```

Regenerating the data replaces `math_data.txt`; retraining replaces the model checkpoint with the best checkpoint for that dataset. Keep a copy of a model if you want to compare experiments.

## Project files

- `generate_data.py` — creates known question/answer examples.
- `model.py` — the neural-network core and greedy character generation.
- `train.py` — character vocabulary, train/validation split, batches, loss, gradient updates, and checkpointing.
- `ask.py` — loads the saved weights and asks the model to continue a prompt.

## What this model can and cannot do

A million random equations is not automatically better than a smaller dataset. Start small so you can run experiments quickly. This model learns statistical patterns in the examples; it is not explicitly taught the addition algorithm, and it may memorize patterns without generalizing. If you need guaranteed correct arithmetic, use actual arithmetic code. If your goal is to study learning, compare the model against unseen test examples and record exact-match accuracy.
