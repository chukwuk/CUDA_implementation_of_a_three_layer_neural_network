import torch
import torch.nn as nn
import torch.optim as optim
import os
import subprocess

#print(torch.cuda.is_available())
if (torch.cuda.is_available()):
   print("CUDA is available") 
else: 
   print("CUDA is not available") 

#x = torch.rand(5, 3)
#print(x)




class SimpleNN(nn.Module):
    def __init__(self, InputNumOfFeatures, numNeuronLayer1, numNeuronLayer2, numNeuronLayer3, custom_variable):
        super(SimpleNN, self).__init__()
        self.custom_variable = custom_variable
        self.linear1 = nn.Linear(InputNumOfFeatures, numNeuronLayer1) # Input size, output size 
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(numNeuronLayer1, numNeuronLayer2) # Input size, output size 
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(numNeuronLayer2, numNeuronLayer3) # Input size, output size 
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.sigmoid(x)
        return x


my_var = "a specific configuration"

print("\n")
numData = int(input("number of Data points: "))
InputNumOfFeatures = int(input("Input number of Features: "))
numNeuronLayer1 = int(input("Enter number of Neuron in Layer 1: "))
numNeuronLayer2 = int(input("Enter number of Neuron in Layer 2: "))
numNeuronLayer3 = int(input("Enter number of Neuron in Layer 3: "))


# Create an instance of the model
model = SimpleNN(InputNumOfFeatures, numNeuronLayer1, numNeuronLayer2, numNeuronLayer3, my_var)

fc1_weights = model.linear1.weight
fc1_bias = model.linear1.bias


fc2_weights = model.linear2.weight
fc2_bias = model.linear2.bias

fc3_weights = model.linear3.weight
fc3_bias = model.linear3.bias



# Create a sample input tensor
input_data = torch.randn(numData, InputNumOfFeatures) # Batch size 1, input features 10
target_data = torch.zeros(numData, 1, requires_grad=True) # target data
#target_data[0][0] = 1
with torch.no_grad():
    target_data[0][0] = 1
  






# Perform forward propagation
output = model(input_data)

#output.retain_grad()
#print(input_data)


# 3. Loss Function and Optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.01)




value = 12345.6789

with open("weightbias1.txt", "w") as f:
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
    for row_idx in range(output.shape[0]):
        for col_idx in range(output.shape[1]):
            element = output[row_idx, col_idx]
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



# apply threshold for binary classification
threshold = 0.5
predictions = (output >= threshold).float()

#predictions.retain_grad() 

with open("predictiondata.txt", "w") as f:
    for row_idx in range(predictions.shape[0]):
        for col_idx in range(predictions.shape[1]):
            element = predictions[row_idx, col_idx]
            f.write(f"{element:.4f}  ")

        f.write(f"\n")
    f.write(f"\n")




loss = criterion(output, target_data)

# Backward and optimize
optimizer.zero_grad()  # Clear previous gradients
loss.backward()        # Compute gradients
optimizer.step()       # Update weights



with open("weightbias3_after_prop.txt", "w") as f:
    # Using f-string with fixed-point format specifier and desired precision
    #f.write(f"{value:.4f}\n") 
    for row_idx in range(model.linear3.weight.shape[0]):
        for col_idx in range(model.linear3.weight.shape[1]):
            element = model.linear3.weight[row_idx, col_idx]
            f.write(f"{element:.4f}  ")
        bias = model.linear3.bias[row_idx]
        f.write(f"{bias:.4f}")
        f.write(f"\n")
    f.write(f"\n")




with open("weightbias3_grad.txt", "w") as f:
    # Using f-string with fixed-point format specifier and desired precision
    #f.write(f"{value:.4f}\n") 
    for row_idx in range(model.linear3.weight.grad.shape[0]):
        for col_idx in range(model.linear3.weight.grad.shape[1]):
            element = model.linear3.weight.grad[row_idx, col_idx]
            f.write(f"{element:.4f}  ")
        bias = model.linear3.bias.grad[row_idx]
        f.write(f"{bias:.4f}")
        f.write(f"\n")
    f.write(f"\n")



with open("weightbias2_after_prop.txt", "w") as f:
    # Using f-string with fixed-point format specifier and desired precision
    #f.write(f"{value:.4f}\n") 
    for row_idx in range(model.linear2.weight.shape[0]):
        for col_idx in range(model.linear2.weight.shape[1]):
            element = model.linear2.weight[row_idx, col_idx]
            f.write(f"{element:.4f}  ")
        bias = model.linear2.bias[row_idx]
        f.write(f"{bias:.4f}")
        f.write(f"\n")
    f.write(f"\n")



with open("weightbias2_grad.txt", "w") as f:
    # Using f-string with fixed-point format specifier and desired precision
    #f.write(f"{value:.4f}\n") 
    for row_idx in range(model.linear2.weight.grad.shape[0]):
        for col_idx in range(model.linear2.weight.grad.shape[1]):
            element = model.linear2.weight.grad[row_idx, col_idx]
            f.write(f"{element:.4f}  ")
        bias = model.linear2.bias.grad[row_idx]
        f.write(f"{bias:.4f}")
        f.write(f"\n")
    f.write(f"\n")



with open("weightbias1_after_prop.txt", "w") as f:
    # Using f-string with fixed-point format specifier and desired precision
    #f.write(f"{value:.4f}\n") 
    for row_idx in range(model.linear1.weight.shape[0]):
        for col_idx in range(model.linear1.weight.shape[1]):
            element = model.linear1.weight[row_idx, col_idx]
            f.write(f"{element:.4f}  ")
        bias = model.linear1.bias[row_idx]
        f.write(f"{bias:.4f}")
        f.write(f"\n")
    f.write(f"\n")



with open("weightbias1_grad.txt", "w") as f:
    # Using f-string with fixed-point format specifier and desired precision
    #f.write(f"{value:.4f}\n") 
    for row_idx in range(model.linear1.weight.grad.shape[0]):
        for col_idx in range(model.linear1.weight.grad.shape[1]):
            element = model.linear1.weight.grad[row_idx, col_idx]
            f.write(f"{element:.4f}  ")
        bias = model.linear1.bias.grad[row_idx]
        f.write(f"{bias:.4f}")
        f.write(f"\n")
    f.write(f"\n")


print("\n")
print("Comparing Pytorch results with CUDA results")
result = subprocess.run(["make"], capture_output=True, text=True)
print("\n")
print(result.stdout)
print(result.stderr) 
print("\n")
result1 = subprocess.run(["./main", str(numData), str(InputNumOfFeatures), str(numNeuronLayer1), str(numNeuronLayer2), str(numNeuronLayer3)], capture_output=True, text=True)
print(result1.stdout)
print(result1.stderr) 


#print(predictions.grad)
#print(torch.zeros(1).cuda())
