import torch
import torch.nn as nn
import torch.optim as optim

print(torch.cuda.is_available())
#x = torch.rand(5, 3)
#print(x)


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.linear1 = nn.Linear(10, 5) # Input size 10, output size 5
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(5, 3) # Input size 5, output size 1
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(3, 1) # Input size 5, output size 1
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        x = self.linear1(x)
        xrelu = self.relu(x)
        x2 = self.linear2(xrelu)
        x3 = self.relu2(x2)
        x4 = self.linear3(x3)
        x4.retain_grad()
        x5 = self.sigmoid(x4)
        return (x, xrelu, x2, x3, x4, x5)




# Create an instance of the model
model = SimpleNN()

fc1_weights = model.linear1.weight
fc1_bias = model.linear1.bias
#output = model.linear1
#print(output)


print("\n")
print("The output of the weight and bias of linear 1 \n")
print(fc1_weights)
print(fc1_bias)
#print(fc1_bias.shape)


print("\n")
print("The output of the weight and bias of linear 2 \n")
fc2_weights = model.linear2.weight
fc2_bias = model.linear2.bias
print(fc2_weights)
print(fc2_bias)



print("\n")
print("The output of the weight and bias of linear 3 \n")
fc3_weights = model.linear3.weight
fc3_bias = model.linear3.bias
print(fc3_weights)
print(fc3_bias)


# Create a sample input tensor
input_data = torch.randn(2, 10) # Batch size 1, input features 10
target_data = torch.zeros(2, 1, requires_grad=True) # target data
#target_data[0][0] = 1
#with torch.no_grad():
    #target_data[0][0] = 1
  


print(input_data)


# 3. Loss Function and Optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.01)


# Perform forward propagation
output1, outputrelu, output2, output3, output4, output5 = model(input_data)

print("\n")
print("The output of the linear1\n")
print(output1)

print("\n")
print("The output of the relu\n")
print(outputrelu)

print("\n")
print("The output of the linear2\n")
print(output2)

print("\n")
print("The output of the second relu\n")
print(output3)


print("\n")
print("The output of the linear3\n")
print(output4)


print("\n")
print("The output of the sigmoid\n")
print(output5)



value = 12345.6789

with open("weightbias.txt", "w") as f:
    # Using f-string with fixed-point format specifier and desired precision
    #f.write(f"{value:.4f}\n") 
    for row_idx in range(fc1_weights.shape[0]):
        for col_idx in range(fc1_weights.shape[1]):
            element = fc1_weights[row_idx, col_idx]
            f.write(f"{element:.4f}  ")
        bias = fc1_bias[row_idx]
        f.write(f"{bias:.4f}")

        f.write(f"\n")

        
    f.write(f"\n")


with open("weightbias2.txt", "w") as f:
    for row_idx in range(fc2_weights.shape[0]):
        for col_idx in range(fc2_weights.shape[1]):
            element = fc2_weights[row_idx, col_idx]
            f.write(f"{element:.4f}  ")
        bias = fc2_bias[row_idx]
        f.write(f"{bias:.4f}")

        f.write(f"\n")


with open("weightbias3.txt", "w") as f:
    for row_idx in range(fc3_weights.shape[0]):
        for col_idx in range(fc3_weights.shape[1]):
            element = fc3_weights[row_idx, col_idx]
            f.write(f"{element:.4f}  ")
        bias = fc3_bias[row_idx]
        f.write(f"{bias:.4f}")

        f.write(f"\n")




with open("inputdata.txt", "w") as f:
    for row_idx in range(input_data.shape[0]):
        for col_idx in range(input_data.shape[1]):
            element = input_data[row_idx, col_idx]
            f.write(f"{element:.4f}  ")

        f.write(f"\n")
    f.write(f"\n")


with open("outputdata.txt", "w") as f:
    for row_idx in range(output5.shape[0]):
        for col_idx in range(output5.shape[1]):
            element = output5[row_idx, col_idx]
            f.write(f"{element:.4f}  ")

        f.write(f"\n")
    f.write(f"\n")


with open("targetdata.txt", "w") as f:
    for row_idx in range(target_data.shape[0]):
        for col_idx in range(target_data.shape[1]):
            element = target_data[row_idx, col_idx]
            f.write(f"{element:.4f}  ")

        f.write(f"\n")
    f.write(f"\n")


#retaing gradient
output5.retain_grad()
#output4.retain_grad()

# apply threshold for binary classification
threshold = 0.5
predictions = (output5 >= threshold).float()

#predictions.retain_grad() 

with open("predictiondata.txt", "w") as f:
    for row_idx in range(predictions.shape[0]):
        for col_idx in range(predictions.shape[1]):
            element = predictions[row_idx, col_idx]
            f.write(f"{element:.4f}  ")

        f.write(f"\n")
    f.write(f"\n")




loss = criterion(predictions, target_data)

# Backward and optimize
optimizer.zero_grad()  # Clear previous gradients
loss.backward()        # Compute gradients
optimizer.step()       # Update weights


for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.data}")

print("\n")
print("gradient of the output")
print(output4.grad)

print("Gradients for fc3.weight:")
print(model.linear3.weight.grad)

print("\nGradients for fc3.bias:")
print(model.linear3.bias.grad)


#print(predictions.grad)
#print(torch.zeros(1).cuda())
