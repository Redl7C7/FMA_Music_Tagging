# Import the library
import torch

# Create a tensor of random numbers of size 3x4
T = torch.randn(3,4)
print("Original Tensor T:\n", T)

# Get the data type of above tensor
data_type = T.dtype

# Print the data type of the tensor
print("Data type of tensor T:\n", data_type)