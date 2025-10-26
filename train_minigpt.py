import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import tiktoken
import wandb
from torch.utils.tensorboard import SummaryWriter

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DIR"] = "./wandb_local"

BLOCK_SIZE = 512
BATCH_SIZE = 16
BASE_LR = 6e-4
WARMUP_STEPS = 2000
MAX_STEPS = 100000
CKPT_DIR = "checkpoints_large"
WANDB_PROJECT = "minigpt_local"
TB_LOGDIR = "./tensorboard_local"

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(TB_LOGDIR, exist_ok=True)

# ===================== WandB & TensorBoard =====================
wandb.init(
    project=WANDB_PROJECT,
    mode="offline",
    config={
        "batch_size": BATCH_SIZE,
        "lr": BASE_LR,
        "warmup_steps": WARMUP_STEPS,
        "max_steps": MAX_STEPS,
        "block_size": BLOCK_SIZE,
    },
)
writer = SummaryWriter(log_dir=TB_LOGDIR)


# ===================== Dataset =====================
class BPEDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=1024):
        print("Tokenizing dataset...")
        ids = []
        for t in tqdm(texts):
            ids.extend(tokenizer.encode(t))
        self.data = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


# ===================== GPT Block =====================
class GPTBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(
            n_embd, n_head, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        T = x.size(1)
        attn_mask = self.mask[:T, :T] == 0
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=attn_mask)[0]
        x = x + self.ff(self.ln2(x))
        return x


# ===================== MiniGPT =====================
class MiniGPT(nn.Module):
    def __init__(
        self, vocab_size, block_size=512, n_layer=6, n_head=8, n_embd=512, dropout=0.1
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[GPTBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.7, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            topk_vals, topk_idx = torch.topk(logits, k=top_k)
            probs = F.softmax(topk_vals, dim=-1)
            next_idx = topk_idx.gather(-1, torch.multinomial(probs, 1))
            idx = torch.cat((idx, next_idx), dim=1)
        return idx


# ===================== Utils =====================
def get_lr(step):
    if step < WARMUP_STEPS:
        return BASE_LR * step / WARMUP_STEPS
    return BASE_LR * 0.5 * (1 + np.cos(np.pi * step / MAX_STEPS))


# ===================== Main =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default="1%")
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from checkpoint"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    print(f"Loading dataset 'Skylion007/openwebtext' subset={args.subset}")
    ds = load_dataset("Skylion007/openwebtext", split=f"train[:{args.subset}]")
    texts = ds["text"]

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = BPEDataset(texts, tokenizer, block_size=BLOCK_SIZE)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )

    model = MiniGPT(vocab_size=tokenizer.n_vocab, block_size=BLOCK_SIZE).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler()
    best_loss = float("inf")
    start_step = 0

    ckpt_path = os.path.join(CKPT_DIR, "latest.pt")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"]
        best_loss = ckpt["best_loss"]
        print(f"Resumed from step {start_step}, best_loss={best_loss:.4f}")

    print("Starting training...")
    for step, (x, y) in enumerate(tqdm(loader, total=args.max_steps)):
        global_step = start_step + step
        if global_step >= args.max_steps:
            break
        x, y = x.to(device), y.to(device)
        lr = get_lr(global_step)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        # ===== Logging =====
        if global_step % 10 == 0:
            wandb.log({"loss": loss.item(), "lr": lr, "step": global_step})
            writer.add_scalar("Loss/train", loss.item(), global_step)
            writer.add_scalar("LearningRate", lr, global_step)
        if global_step % 100 == 0:
            print(f"[Step {global_step}] loss={loss.item():.4f} lr={lr:.6f}")

        # ===== Checkpoint =====
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best_model.pt"))
        if global_step % 500 == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": global_step,
                    "best_loss": best_loss,
                },
                ckpt_path,
            )

        # ===== Sample generation =====
        if global_step % 1000 == 0:
            model.eval()
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            out = (
                model.module.generate(context)
                if isinstance(model, nn.DataParallel)
                else model.generate(context)
            )
            sample = tokenizer.decode(out[0].tolist())
            wandb.log({"generated_text": sample})
            writer.add_text("Generated_Text", sample, global_step)
            print("\nGenerated Text:\n", sample)
            model.train()

    print("Training complete.")
    writer.close()
    wandb.finish()


if __name__ == "__main__":
    main()
