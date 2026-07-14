import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# 1. Load same dataset mapping (IMPORTANT)
# -------------------------
text = open("data.txt", "r", encoding="utf-8").read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])


# -------------------------
# 2. Model definition (MUST match training exactly)
# -------------------------
block_size = 128

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(128, 128, bias=False)
        self.query = nn.Linear(128, 128, bias=False)
        self.value = nn.Linear(128, 128, bias=False)
        self.tril = torch.tril(torch.ones(block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
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

    def forward(self, idx):
        B, T = idx.shape

        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T))

        x = tok + pos
        x = self.blocks(x)
        x = self.ln(x)

        logits = self.head(x)
        return logits


# -------------------------
# 3. Load trained model
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = GPT().to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()


# -------------------------
# 4. Generate function
# -------------------------
def generate(max_new_tokens=500, temperature=0.9):
    idx = torch.zeros((1, 1), dtype=torch.long).to(device)

    for _ in range(max_new_tokens):
        logits = model(idx)

        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)

        next_id = torch.multinomial(probs, 1)

        idx = torch.cat([idx, next_id], dim=1)

    return decode(idx[0].tolist())


# -------------------------
# 5. Run generation
# -------------------------
story = generate(max_new_tokens=800, temperature=0.9)

lines = story.split("\n")[:30]

print("\n".join(lines))
