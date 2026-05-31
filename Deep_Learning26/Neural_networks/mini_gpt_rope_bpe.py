import os
import time
import torch
import torch.nn as nn
from tokenizers import ByteLevelBPETokenizer

# ==========================================
# 1. Hyperparameters (Optimized for local CPU)
# ==========================================
batch_size = 16       # How many independent sequences to process in parallel
block_size = 16       # Maximum context length (reduced slightly to fit tiny dataset gracefully)
embed_dim = 64        # Dimensionality of embeddings (must be even for RoPE)
num_layers = 2        # Number of transformer blocks
learning_rate = 1e-3  # Learning rate for Adam optimizer
max_iters = 1000      # Number of training steps
eval_interval = 200   # How often to check loss and print sample generations
device = "cpu"        # Specifically targeting CPU

torch.manual_seed(42)

# ==========================================
# 2. Dataset Preparation & BPE Tokenizer
# ==========================================
# An expanded sample story to ensure sufficient data points for BPE token slices.
dataset_text = """Once upon a time, there was a tiny robot named PyTorch.
PyTorch loved to learn how to speak.
Every day, the robot practiced reading books and writing sentences.
Slowly but surely, PyTorch became very good at predicting the next word.
The little robot lived happily ever after, learning and dreaming in the neural network.
It was a beautiful journey of discovery and learning for the robot.
Every single parameter was tuned with care and precision to make it smart.
In the world of deep learning, PyTorch found its true purpose and joy."""

data_filename = "tiny_story.txt"
if not os.path.exists(data_filename):
    with open(data_filename, 'w', encoding='utf-8') as f:
        f.write(dataset_text)

# Read the file
with open(data_filename, 'r', encoding='utf-8') as f:
    text = f.read()

# Initialize and train a Byte-Level BPE Tokenizer
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=[data_filename],
    vocab_size=150,  # Small vocabulary size suitable for a tiny local story
    min_frequency=1,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)

vocab_size = tokenizer.get_vocab_size()

# Encoder and decoder helper functions using Byte-Level BPE
encode = lambda s: tokenizer.encode(s).ids
decode = lambda l: tokenizer.decode(l)

# Convert dataset to token IDs
data_tensor = torch.tensor(encode(text), dtype=torch.long)
n = int(0.8 * len(data_tensor))  # 80/20 Train/Validation Split
train_data = data_tensor[:n]
val_data = data_tensor[n:]

# Data batch generator
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ==========================================
# 3. Rotary Position Embedding (RoPE) Helpers
# ==========================================

def precompute_freqs_cis(dim, end, theta=10000.0):
    """
    Precomputes the cosine and sine components for RoPE.
    Assumes dimension (dim) is even.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (end, dim/2)
    freqs = torch.repeat_interleave(freqs, 2, dim=-1)  # Duplicate to match (end, dim)
    return freqs.cos(), freqs.sin()


def apply_rotary_emb(x, cos, sin):
    """
    Applies rotary position embeddings to query or key tensors of shape (B, T, C).
    """
    T = x.size(1)
    # Slice the precomputed tables to match sequence length T and align dimensions
    cos = cos[:T, :].unsqueeze(0)  # (1, T, C)
    sin = sin[:T, :].unsqueeze(0)  # (1, T, C)

    # Perform the rotation on adjacent element pairs
    x_rotated = torch.empty_like(x)
    x_rotated[..., 0::2] = -x[..., 1::2]
    x_rotated[..., 1::2] = x[..., 0::2]
    return x * cos + x_rotated * sin


# ==========================================
# 4. Model Architecture Components
# ==========================================

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.size()
        
        Q = self.W_q(x)  # (B, T, C)
        K = self.W_k(x)  # (B, T, C)
        V = self.W_v(x)  # (B, T, C)

        # Apply rotary positional transformations to Queries and Keys
        Q = apply_rotary_emb(Q, cos, sin)
        K = apply_rotary_emb(K, cos, sin)

        # Causal Attention Score computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (C ** 0.5)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, T, T)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, V)


class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, num_layers):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        
        # Precompute RoPE tables for the maximum block size
        cos, sin = precompute_freqs_cis(embed_dim, block_size)
        self.register_buffer("cos_table", cos)
        self.register_buffer("sin_table", sin)

        # Use nn.ModuleList to pass positional parameters dynamically
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Token embedding lookup (no learning-based position embedding table used)
        x = self.token_embedding_table(idx)  # (B, T, embed_dim)
        
        # Gather correct portion of precomputed RoPE matrices
        cos = self.cos_table[:T]
        sin = self.sin_table[:T]
        
        # Propagate embeddings and RoPE components through blocks
        for block in self.blocks:
            x = block(x, cos, sin)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss = nn.CrossEntropyLoss()(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ==========================================
# 5. Initialization and Training Loop
# ==========================================

# Initialize the model
model = MiniGPT(vocab_size, embed_dim, block_size, num_layers).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Device in use: {device}")
print(f"Dataset BPE vocabulary size: {vocab_size}")
print(f"Total model parameters: {sum(p.numel() for p in model.parameters())}")
print("Starting training process...\n")


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(50)
        for k in range(50):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Training Loop
start_time = time.time()
for iteration in range(max_iters + 1):

    if iteration % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iteration:4d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")
        
        # Quick validation inference during training (index 0 corresponds to '<s>' start token)
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        sampled_indices = model.generate(context, max_new_tokens=25)[0].tolist()
        print(f"--- Generated Sample ---\n{decode(sampled_indices)}\n------------------------\n")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")

# ==========================================
# 6. Final Inference
# ==========================================
print("\n=== Final Text Generation (From Custom Prompt) ===")
prompt = "Once "
prompt_encoded = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
generated_idx = model.generate(prompt_encoded, max_new_tokens=40)[0].tolist()
print(decode(generated_idx))


# # Verify that encoding + decoding is lossless
# sample = "Once upon a time, PyTorch loves learning."
# ids = encode(sample)
# assert decode(ids) == sample, "Tokenizer round‑trip failed!"

# # Verify the new embedding shape
# model = MiniGPT(vocab_size, embed_dim=64, block_size=32, num_layers=2).to(device)
# dummy = torch.tensor([ids[:32]], dtype=torch.long, device=device)  # pad/truncate to block_size
# logits, _ = model(dummy)                                            # (1,32,vocab_size)
# print("logits shape →", logits.shape)