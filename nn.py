import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Assuming you have a regression task for X and Y coordinates
num_output_neurons = 2  # Two output neurons for X and Y

# Simple CNN model for regression
class SimpleCNNRegression(nn.Module):
    def __init__(self):
        super(SimpleCNNRegression, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * (379 // 4) * (596 // 4), 128)
        self.fc2 = nn.Linear(128, num_output_neurons)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create an instance of the regression model
model_regression = SimpleCNNRegression()

# Define a loss function suitable for regression (Mean Squared Error)
criterion_regression = nn.MSELoss()
optimizer_regression = optim.Adam(model_regression.parameters(), lr=0.001)

image_np = np.load('image_array.npy')
coords_np = np.load('coords_array.npy')[:352]

# Convert your numpy array to a PyTorch tensor
# Assuming your numpy array is named 'image_array'
image_tensor = torch.tensor(image_np, dtype=torch.float32)

# Concatenate X and Y tensors to form the labels tensor
coords_tensor = torch.tensor(coords_np, dtype=torch.float32)

# Create a DataLoader for handling batches
dataset_regression = TensorDataset(image_tensor, coords_tensor)
dataloader_regression = DataLoader(dataset_regression, batch_size=32, shuffle=True)

# Training loop for regression
num_epochs_regression = 10
for epoch in range(num_epochs_regression):
    for inputs, labels in dataloader_regression:
        optimizer_regression.zero_grad()
        outputs = model_regression(inputs)
        loss = criterion_regression(outputs, labels)
        loss.backward()
        optimizer_regression.step()

    print(f'Epoch [{epoch+1}/{num_epochs_regression}], Loss: {loss.item()}')

# Save the trained regression model if needed
torch.save(model_regression.state_dict(), 'simple_cnn_regression_model.pth')
