import torch
import torch.nn as nn

# Set device to CPU
device = torch.device("cpu")

# =====================================================================
# SIMULATED LLM CONFIGURATION
# =====================================================================
VOCAB_SIZE = 50257      # GPT-2 standard vocabulary size
HIDDEN_DIM = 768        # GPT-2 base model embedding dimension (d_model)
BOTTLENECK_DIM = 256    # A smaller projection dimension (e.g., for an adapter/LoRA layer)
PADDING_IDX = 0         # Token ID reserved for padding

print(f"LLM Configuration:")
print(f" - Vocabulary Size: {VOCAB_SIZE}")
print(f" - Hidden Dimension: {HIDDEN_DIM}")
print(f" - Bottleneck/Reduced Dimension: {BOTTLENECK_DIM}\n")


# =====================================================================
# STEP 1: LLM INPUT PADDING & ATTENTION MASKING
# =====================================================================
print("--- Step 1: Padding and Masking ---")

# Imagine 3 incoming text sequences of different token lengths
doc1 = torch.tensor([101, 2054, 2003, 1037, 102], dtype=torch.long)  # Length: 5
doc2 = torch.tensor([101, 2300, 102], dtype=torch.long)              # Length: 3
doc3 = torch.tensor([101, 102], dtype=torch.long)                    # Length: 2

sequences = [doc1, doc2, doc3]

# Pad the sequences to the length of the longest sequence (5)
padded_input_ids = torch.nn.utils.rnn.pad_sequence(
    sequences, 
    batch_first=True, 
    padding_value=PADDING_IDX
)

# In LLMs, we also generate an "Attention Mask" (1 for real tokens, 0 for pad tokens)
# to prevent the model from performing self-attention on padding tokens.
attention_mask = (padded_input_ids != PADDING_IDX).long()

print("Padded Input IDs (Token Indices):")
print(padded_input_ids)
print("\nAttention Mask:")
print(attention_mask)

# DIMENSION CHANGE:
# Before padding: List of 1D tensors with shapes [5], [3], [2]
# After padding:  2D tensor of shape [3, 5] -> [batch_size, max_seq_len]
# Attention Mask: 2D tensor of shape [3, 5] -> [batch_size, max_seq_len]
print(f"\nShape of padded input: {list(padded_input_ids.shape)}")
print(f"Shape of attention mask: {list(attention_mask.shape)}\n")


# =====================================================================
# STEP 2: SPARSE-TO-DENSE VIA THE LLM EMBEDDING LAYER
# =====================================================================
print("--- Step 2: Token IDs to Dense Embedding Matrix ---")

# In LLMs, passing a token ID (e.g., 2054) is conceptually equivalent to multiplying 
# a highly sparse one-hot vector [1, VOCAB_SIZE] by an embedding matrix [VOCAB_SIZE, HIDDEN_DIM].
# nn.Embedding performs this lookup operation efficiently.
embedding_layer = nn.Embedding(
    num_embeddings=VOCAB_SIZE, 
    embedding_dim=HIDDEN_DIM, 
    padding_idx=PADDING_IDX
)

# Convert the sparse token IDs into dense representations
dense_embeddings = embedding_layer(padded_input_ids)

# DIMENSION CHANGE:
# Before: [3, 5] -> [batch_size, max_seq_len]
# After:  [3, 5, 768] -> [batch_size, max_seq_len, hidden_dim]
print(f"Shape of dense embedding matrix: {list(dense_embeddings.shape)}")


# =====================================================================
# STEP 3: DIMENSIONALITY REDUCTION (BOTTLENECK / ADAPTER PROJECTION)
# =====================================================================
print("\n--- Step 3: Reducing Embedding Dimension ---")

# Inside an LLM, we might reduce the hidden representation dimension to save compute 
# or to fit into a low-rank adapter (e.g., LoRA). We use a linear projection.
reduction_projection = nn.Linear(
    in_features=HIDDEN_DIM, 
    out_features=BOTTLENECK_DIM, 
    bias=False
)

# Project the embeddings from HIDDEN_DIM (768) down to BOTTLENECK_DIM (256)
reduced_embeddings = reduction_projection(dense_embeddings)

# DIMENSION CHANGE:
# Before: [3, 5, 768] -> [batch_size, max_seq_len, hidden_dim]
# After:  [3, 5, 256] -> [batch_size, max_seq_len, bottleneck_dim]
print(f"Shape of reduced embedding matrix: {list(reduced_embeddings.shape)}")


# =====================================================================
# STEP 4: APPLYING ATTENTION MASK TO THE REDUCED REPRESENTATIONS
# =====================================================================
print("\n--- Step 4: Applying Attention Mask ---")

# In standard LLM layers, we make sure that padding values do not carry 
# projected information forward. We zero out the masked positions.
# We expand the mask from [3, 5] to [3, 5, 1] so it broadcasts across the bottleneck dimension.
expanded_mask = attention_mask.unsqueeze(-1)  # Shape: [3, 5, 1]

masked_reduced_embeddings = reduced_embeddings * expanded_mask

# DIMENSION CHANGE:
# Before Masking: [3, 5, 256] -> [batch_size, max_seq_len, bottleneck_dim]
# After Masking:  [3, 5, 256] -> [batch_size, max_seq_len, bottleneck_dim] (values at pad indices are now 0.0)
print(f"Shape after masking: {list(masked_reduced_embeddings.shape)}")

# Print row index 1 (originally length 3, now length 5 with 2 padded/zeroed positions)
print("\nExample of masked sample in batch (note the last two token vectors are zeroed out):")
print(masked_reduced_embeddings[1])