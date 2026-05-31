import os
import time
import torch
import torch.nn as nn

# ==========================================
# 1. Hyperparameters (Optimized for local CPU)
# ==========================================
batch_size = 16       # How many independent sequences to process in parallel
block_size = 32       # Maximum context length (window size)
embed_dim = 64        # Dimensionality of character embeddings
num_layers = 2        # Number of transformer blocks
learning_rate = 1e-3  # Learning rate for Adam optimizer
max_iters = 1500      # Number of training steps
eval_interval = 300   # How often to check loss and print sample generations
device = "cpu"        # Specifically targeting CPU as requested

torch.manual_seed(42)

# ==========================================
# 2. Dataset Preparation
# ==========================================
# A sample story used as the learning dataset.
# The code writes this to a local text file to mimic loading custom data.
dataset_text = """Once upon a time, there was a tiny robot named PyTorch.
PyTorch loved to learn how to speak.
Every day, the robot practiced reading books and writing sentences.
Slowly but surely, PyTorch became very good at predicting the next word.
The little robot lived happily ever after, learning and dreaming in the neural network."""

data_filename = "tiny_story.txt"
if not os.path.exists(data_filename):
    with open(data_filename, 'w', encoding='utf-8') as f:
        f.write(dataset_text)

# Read the file
with open(data_filename, 'r', encoding='utf-8') as f:
    text = f.read()

# Character-level tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Encoder and decoder functions
encode = lambda s: [char_to_idx[c] for c in s]
decode = lambda l: ''.join([idx_to_char[i] for i in l])

# Split dataset into Train and Validation sets
data_tensor = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data_tensor))
train_data = data_tensor[:n]
val_data = data_tensor[n:]

# Data batch generator
def get_batch(split):
    data = train_data if split == 'train' else val_data
    # Avoid picking an index that would overshoot the dataset boundaries
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ==========================================
# 3. Model Architecture Components
# ==========================================

class CausalSelfAttention(nn.Module):
    """
    An adaptation of your SelfAttention block. 
    It includes causal masking to prevent looking at future tokens.
    """
    def __init__(self, embed_dim):
        super().__init__()
        # Projections for Q, K, and V
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        B, T, C = x.size() # Batch size, Sequence length, Embedding dimension
        
        # 1. Project inputs to Query, Key, and Value
        Q = self.W_q(x) # (B, T, C)
        K = self.W_k(x) # (B, T, C)
        V = self.W_v(x) # (B, T, C)

        # 2. Score calculation with scaling
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (C ** 0.5) # (B, T, T)
        
        # 3. Causal Masking: fill upper-triangular elements with -infinity.
        # This forces the attention weights for future elements to evaluate to 0 during Softmax.
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, T, T)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # 4. Softmax step
        weights = torch.softmax(scores, dim=-1) # (B, T, T)

        # 5. Output calculation
        return torch.matmul(weights, V) # (B, T, C)


class FeedForward(nn.Module):
    """
    A simple linear layer followed by non-linearity and projection.
    """
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
    """
    A single Transformer block combining normalization, self-attention, and feedforward networks.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim)

    def forward(self, x):
        # Utilizing pre-normalization layers and residual connections
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """
    The full Language Model architecture containing embeddings, blocks, and language model head.
    """
    def __init__(self, vocab_size, embed_dim, block_size, num_layers):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding_table = nn.Embedding(block_size, embed_dim)
        
        # Stack multiple Transformer blocks
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim) # Final LayerNorm
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Retrieve both token and positional representations
        tok_emb = self.token_embedding_table(idx) # (B, T, embed_dim)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, embed_dim)
        
        x = tok_emb + pos_emb # Shape: (B, T, embed_dim)
        x = self.blocks(x)     # Shape: (B, T, embed_dim)
        x = self.ln_f(x)      # Shape: (B, T, embed_dim)
        logits = self.lm_head(x) # Shape: (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Reshape tensors to conform to PyTorch's CrossEntropyLoss shape criteria
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss = nn.CrossEntropyLoss()(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # Generates successive characters autoregressively
        for _ in range(max_new_tokens):
            # Crop current context window to the max block size supported
            idx_cond = idx[:, -self.block_size:]
            
            # Predict the next tokens
            logits, _ = self(idx_cond)
            
            # Extract prediction for only the final step
            logits = logits[:, -1, :] # (B, vocab_size)
            
            probs = torch.softmax(logits, dim=-1) # Calculate probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # Sample from output distribution
            
            # Append selected token sequence for the next iteration step
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ==========================================
# 4. Initialization and Training Loop
# ==========================================

# Initialize the model
model = MiniGPT(vocab_size, embed_dim, block_size, num_layers).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Device in use: {device}")
print(f"Dataset unique characters size: {vocab_size}")
print(f"Total model parameters: {sum(p.numel() for p in model.parameters())}")
print("Starting training process...\n")

# Evaluation helper to estimate loss over multiple batches
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

    # Every so often, evaluate model's current performance
    if iteration % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iteration:4d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")
        
        # Run a quick validation inference during training
        context = torch.zeros((1, 1), dtype=torch.long, device=device) # Start token (index 0)
        sampled_indices = model.generate(context, max_new_tokens=60)[0].tolist()
        print(f"--- Generated Sample ---\n{decode(sampled_indices)}\n------------------------\n")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Forward pass
    logits, loss = model(xb, yb)

    # Backward pass and optimization step
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")

# ==========================================
# 5. Final Inference
# ==========================================
print("\n=== Final Text Generation (From Custom Prompt) ===")
# Prompt the model with "Once "
prompt = "Once "
# Encode prompt, shape (1, T)
prompt_encoded = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
generated_idx = model.generate(prompt_encoded, max_new_tokens=150)[0].tolist()
print(decode(generated_idx))