import torch
import torch.nn as nn

# 1. Define a simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNet()

# 2. Create dummy input and target
input_data = torch.randn(1, 10)
target = torch.randn(1, 1)

# 3. Perform forward pass and calculate loss
output = model(input_data)
loss = torch.mean((output - target)**2) # Example: Mean squared error

# 4. Perform backward pass to compute gradients
loss.backward()

# 5. Access and print the gradients
print("Gradients for fc1.weight:")
print(model.fc1.weight.grad)

print("\nGradients for fc1.bias:")
print(model.fc1.bias.grad)

print("\nGradients for fc2.weight:")
print(model.fc2.weight.grad)

print("\nGradients for fc2.bias:")
print(model.fc2.bias.grad)

# Alternatively, iterate through all parameters
print("\nGradients for all parameters:")
#for name, param in model.named_parameters():
 #   if param.grad is not None:
  #      print(f"{name}.grad:\n{param.grad}")
  #  else:
   #     print(f"{name}.grad: None (No gradient computed)")
