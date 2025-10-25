import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define the Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 2. Instantiate Model, Loss Function, and Optimizer
input_size = 10
hidden_size = 20
output_size = 3  # For 3 classes
model = SimpleNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()  # Entropy loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example data
inputs = torch.randn(64, input_size)  # Batch size 64
labels = torch.randint(0, output_size, (64,))

# Training loop (simplified for one iteration)
# 3. Forward Propagation
outputs = model(inputs)

# 4. Calculate Loss
loss = criterion(outputs, labels)

# 5. Backward Propagation
optimizer.zero_grad()  # Clear previous gradients
loss.backward()        # Compute gradients

# 6. Parameter Optimization
optimizer.step()       # Update parameters with Adam
