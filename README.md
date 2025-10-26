# MiniGPT: Lightweight GPT-like Language Model Training Framework

## Overview

**MiniGPT** is a simplified implementation of the GPT (Generative Pretrained Transformer) architecture using PyTorch.
It demonstrates how to train a Transformer-based text generation model from scratch, featuring:

- **OpenWebText** dataset loading
- **BPE (tiktoken GPT-2)** tokenization
- **WandB** and **TensorBoard** integration for logging and visualization

This project is ideal for **researchers, students, and practitioners** who want to understand GPT training and text generation at a code level.

---

## Installation & Dependencies

### Requirements

- **Python** ≥ 3.8
- **CUDA** ≥ 11.3 (optional, for GPU acceleration)

### Install dependencies

```bash
pip install torch datasets==3.6.0 accelerate transformers==4.57.1 tqdm matplotlib scipy tiktoken wandb tensorboard
```

Optional (for multi-GPU support):

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
```

---

## Usage

### 1. Start Training

```bash
python train_minigpt.py --subset 1% --max_steps 50000
```

#### Command-line Arguments

| Argument      | Description                                            | Default  |
| ------------- | ------------------------------------------------------ | -------- |
| `--subset`    | Dataset portion to use from OpenWebText (e.g., `"1%"`) | `"1%"`   |
| `--max_steps` | Maximum number of training steps                       | `100000` |
| `--resume`    | Resume from the latest checkpoint                      | `False`  |

Logs are saved to:

- **WandB (offline mode)** → `./wandb_local`
- **TensorBoard** → `./tensorboard_local`

#### View logs with TensorBoard

```bash
tensorboard --logdir ./tensorboard_local
```

---

### 2. Model Architecture

MiniGPT consists of:

- **`GPTBlock`** – Implements multi-head self-attention, feed-forward layers, and residual connections.
- **`MiniGPT`** – Stacks multiple `GPTBlock`s with token embeddings, positional encodings, and a linear output head.
- **`BPEDataset`** – Prepares tokenized sequences and training samples `(x, y)` for autoregressive prediction.

---

### 3. Generate Text from a Trained Model

```python
import torch
from train_minigpt import MiniGPT
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
device = "cuda" if torch.cuda.is_available() else "cpu"

model = MiniGPT(vocab_size=tokenizer.n_vocab).to(device)
model.load_state_dict(torch.load("checkpoints_large/best_model.pt", map_location=device))
model.eval()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
output = model.generate(context, max_new_tokens=200)
print(tokenizer.decode(output[0].tolist()))
```

---

## License

You are free to use, modify, and distribute it, provided that the copyright notice is retained.
