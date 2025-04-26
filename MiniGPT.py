import torch
import torch.nn as nn
import torch.optim as optim

class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_size):
        super(SimpleGPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_size
            ) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        return self.fc_out(x)

def save_model(model, filepath="gpt_model.pth"):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(model, filepath="gpt_model.pth"):
    model.load_state_dict(torch.load(filepath))
    model.eval()
    print(f"Model loaded from {filepath}")


def train_model(model, data, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, model.vocab_size), data.view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

        