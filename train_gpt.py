# 1. imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# 2. load dataset
text = open("data.txt", "r", encoding="utf-8").read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

# 3. train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# 4. batch loader
block_size = 128
batch_size = 32

def get_batch(split="train"):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x, y

# 5. model
class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(128, 128, bias=False)
        self.query = nn.Linear(128, 128, bias=False)
        self.value = nn.Linear(128, 128, bias=False)
        self.tril = torch.tril(torch.ones(block_size, block_size))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        return wei @ v


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = Head()
        self.ff = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.ln1 = nn.LayerNorm(128)
        self.ln2 = nn.LayerNorm(128)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, 128)
        self.position_embedding = nn.Embedding(block_size, 128)

        self.blocks = nn.Sequential(
            Block(),
            Block(),
            Block()
        )

        self.ln = nn.LayerNorm(128)
        self.head = nn.Linear(128, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape

        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T))

        x = tok + pos
        x = self.blocks(x)
        x = self.ln(x)

        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                targets.view(-1)
            )

        return logits, loss

# 6. training
device = "cuda" if torch.cuda.is_available() else "cpu"

model = GPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for step in range(5000):
    xb, yb = get_batch()
    xb, yb = xb.to(device), yb.to(device)

    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(step, loss.item())

# 7. generation
def generate(max_new_tokens=500):
    model.eval()
    # Start with a batch of 1, containing a single 0 token (usually the newline or null token)
    idx = torch.zeros((1, 1), dtype=torch.long).to(device)

    # Disable gradient calculation to save memory and speed up generation
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # --- FIX: Crop idx to the last block_size tokens so it doesn't overflow ---
            idx_cond = idx[:, -block_size:]
            
            # Pass the safely cropped context into the model
            logits, _ = model(idx_cond)
            
            # Focus only on the very last time step
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            # Sample the next token from the distribution
            next_id = torch.multinomial(probs, 1)
            
            # Append the sampled index to the running sequence
            idx = torch.cat([idx, next_id], dim=1)

    return decode(idx[0].tolist())

print(generate())

