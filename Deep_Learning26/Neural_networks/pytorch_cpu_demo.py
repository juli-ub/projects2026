import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. CUSTOM PYTORCH AUTOGRAD FUNCTION
# ==========================================
class CustomSwishFunction(torch.autograd.Function):
    """
    A custom autograd function representing f(x) = x * sigmoid(x).
    We manually define both the forward and backward passes.
    """
    @staticmethod
    def forward(ctx, x):
        # Save input tensor for the backward derivative pass
        ctx.save_for_backward(x)
        sigmoid_x = torch.sigmoid(x)
        return x * sigmoid_x

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve input tensor
        x, = ctx.saved_tensors
        sigmoid_x = torch.sigmoid(x)
        
        # Derivative: d/dx [x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        derivative = sigmoid_x + x * sigmoid_x * (1.0 - sigmoid_x)
        
        # Chain rule: return gradient received from upstream times our local derivative
        return grad_output * derivative

# Helper interface to call the custom function
custom_swish = CustomSwishFunction.apply


# ==========================================
# 2. TOY DATA & CUSTOM DATASET
# ==========================================
# Raw text data for binary sentiment classification (1 = Positive, 0 = Negative)
train_raw_data = [
    ("i love this movie", 1.0),
    ("this was an awesome experience", 1.0),
    ("absolutely fantastic performance", 1.0),
    ("i hated this film", 0.0),
    ("this is a terrible movie", 0.0),
    ("worst acting ever", 0.0)
]

fine_tune_raw_data = [
    ("great show and highly recommended", 1.0),
    ("the plot was boring and bad", 0.0)
]

# Simple tokenization and vocabulary construction
all_texts = [text for text, _ in train_raw_data + fine_tune_raw_data]
vocab = {"<pad>": 0, "<unk>": 1}
for text in all_texts:
    for word in text.lower().split():
        if word not in vocab:
            vocab[word] = len(vocab)

MAX_SEQUENCE_LENGTH = 5

def encode_text(text, vocabulary, max_len):
    """Tokenizes and pads text into a static-length list of indices."""
    tokens = text.lower().split()
    encoded = [vocabulary.get(word, vocabulary["<unk>"]) for word in tokens]
    # Handle padding or truncation
    if len(encoded) < max_len:
        encoded += [vocabulary["<pad>"]] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]
    return torch.tensor(encoded, dtype=torch.long)

class TextClassificationDataset(Dataset):
    """Custom PyTorch Dataset representing our text data."""
    def __init__(self, raw_data, vocabulary, max_len):
        self.data = []
        for text, label in raw_data:
            encoded_tensor = encode_text(text, vocabulary, max_len)
            label_tensor = torch.tensor(label, dtype=torch.float32)
            self.data.append((encoded_tensor, label_tensor))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ==========================================
# 3. CUSTOM NN MODULE
# ==========================================
class CustomTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim=1):
        super().__init__()
        # PyTorch Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)  # shape: (batch_size, sequence_length, embedding_dim)
        
        # Custom operation: global average pooling over sequence length
        pooled = torch.mean(embedded, dim=1)  # shape: (batch_size, embedding_dim)
        
        # Dense layer
        x = self.fc1(pooled)
        
        # Call our custom autograd activation function
        x = custom_swish(x)
        
        # Output layer (sigmoid yields boundary values between 0 and 1)
        x = self.fc2(x)
        return torch.sigmoid(x)


# ==========================================
# 4. TRAINING FUNCTION
# ==========================================
def train_model(model, dataloader, epochs=30, lr=0.01):
    print("--- Starting Initial Training ---")
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for texts, labels in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(texts).squeeze(-1)
            loss = criterion(predictions, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {epoch_loss / len(dataloader):.4f}")
    print("Initial Training Complete.\n")


# ==========================================
# 5. INFERENCE FUNCTION
# ==========================================
def run_inference(model, text, vocabulary, max_len):
    model.eval()
    with torch.no_grad():
        encoded = encode_text(text, vocabulary, max_len).unsqueeze(0)  # Add batch dimension
        prediction = model(encoded).item()
        sentiment = "Positive" if prediction >= 0.5 else "Negative"
        print(f"Input: '{text}' -> Score: {prediction:.4f} ({sentiment})")


# ==========================================
# 6. FINE-TUNING FUNCTION
# ==========================================
def fine_tune_model(model, fine_tune_loader, epochs=15, lr=0.001):
    print("--- Starting Fine-Tuning ---")
    print("Freezing the Embedding layer weights to preserve base knowledge...")
    
    # Custom action: Freeze specific weights (fine-tuning pattern)
    for param in model.embedding.parameters():
        param.requires_grad = False
        
    # Only optimize parameters that require gradients
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.BCELoss()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for texts, labels in fine_tune_loader:
            optimizer.zero_grad()
            predictions = model(texts).squeeze(-1)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Fine-Tune Epoch {epoch+1:02d}/{epochs} | Loss: {epoch_loss / len(fine_tune_loader):.4f}")
            
    # Unfreeze parameters back to default state
    for param in model.embedding.parameters():
        param.requires_grad = True
        
    print("Fine-Tuning Complete.\n")


# ==========================================
# 7. EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    # Ensure CPU is explicitly selected
    device = torch.device("cpu")
    print(f"Running pipeline on: {device.type.upper()}\n")

    # Create PyTorch datasets and loaders
    train_dataset = TextClassificationDataset(train_raw_data, vocab, MAX_SEQUENCE_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    fine_tune_dataset = TextClassificationDataset(fine_tune_raw_data, vocab, MAX_SEQUENCE_LENGTH)
    fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=1, shuffle=True)

    # Initialize model
    model = CustomTextClassifier(
        vocab_size=len(vocab),
        embedding_dim=8,
        hidden_dim=4
    ).to(device)

    # 1. Base Training
    train_model(model, train_loader, epochs=25, lr=0.01)

    # 2. Testing Base Model (Inference)
    print("--- Testing Predictions Post-Training ---")
    run_inference(model, "i love this", vocab, MAX_SEQUENCE_LENGTH)
    run_inference(model, "terrible acting", vocab, MAX_SEQUENCE_LENGTH)
    print("")

    # 3. Fine-Tuning on supplementary data
    fine_tune_model(model, fine_tune_loader, epochs=15, lr=0.005)

    # 4. Final Testing (Inference)
    print("--- Testing Predictions Post-Fine-Tuning ---")
    run_inference(model, "highly recommended", vocab, MAX_SEQUENCE_LENGTH)
    run_inference(model, "boring and bad", vocab, MAX_SEQUENCE_LENGTH)