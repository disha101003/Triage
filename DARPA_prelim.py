# Load your dataset into a suitable data structure, such as a Pandas DataFrame.

import pandas as pd

# Read the CSV file into a Pandas DataFrame
file_path = r"TATRCManikinOnlyDataset\Collect 1\D01_G1_vitals_C3.csv"

df = pd.read_csv(file_path)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd

# Assuming df is your Pandas DataFrame

# Extract features and labels
X = torch.tensor(df.drop('Estimated Blood Lossbsp;', axis=1).values, dtype=torch.float32)
y = torch.tensor(df['Estimated Blood Lossbsp;'].values, dtype=torch.float32)

# Define the neural network model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Specify the input, hidden, and output sizes
input_size = X.shape[1]
hidden_size = 20  # Adjust as needed
output_size = 1   # For binary classification

# Create an instance of the NeuralNet model
model = NeuralNet(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Split the data into train and test sets
train_size = int(0.8 * len(X))
test_size = len(X) - train_size
train_dataset, test_dataset = random_split(TensorDataset(X, y), [train_size, test_size])

# Create DataLoader objects
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {total_loss/len(train_loader)}")
