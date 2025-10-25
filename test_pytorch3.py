import torch

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
loss = y + 5

loss.backward() # Computes gradients

print(x.grad) # Output: tensor(6.)
