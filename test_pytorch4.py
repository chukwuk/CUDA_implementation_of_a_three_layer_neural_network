import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define your model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 5) # Input size 10, output size 1
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(5, 1) # Input size 5, output size 1

    def forward(self, x):
         x = self.linear(x)
         x = self.relu(x)
         x = self.linear2(x)
         return x

model = SimpleModel()

# 2. Define your optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Generate some dummy data
input_data = torch.randn(1, 10)
target_data = torch.randn(1, 1)

# Print initial weights (optional)
print("Initial weights:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.data}")

# 3. Perform forward pass
output = model(input_data)

# 4. Calculate the loss
loss = nn.MSELoss()(output, target_data)

# 5. Zero the gradients
optimizer.zero_grad()

# 6. Perform backpropagation
loss.backward()

# 7. Update the weights
optimizer.step()

# 8. Print the new weights
print("\nNew weights after first backpropagation:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.data}")
