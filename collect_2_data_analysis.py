import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

df = pd.read_csv(
    "c:/Users/suddu/Downloads/Python/ML/TATRCManikinOnlyDataset/Collect 1/D01_G1_vitals_C3.csv"
)

# Remove columns with all zeros
df = df.loc[:, (df != 0).any(axis=0)]

data = torch.tensor(df.values)

# Number of data points
print(len(data))  # 151501

# Number of features
print(data.shape[1])  # Updated number of features after removing columns

# Feature names
print(df.columns)

# Strip leading and trailing whitespaces from column names
df.columns = df.columns.str.strip()

plt.plot(df["Elapsed Time (ms)"], df["HR"])
plt.title("Heart Rate Over Time")
plt.xlabel("Time (ms)")
plt.ylabel("HR (bpm)")
plt.show()

# Blood Pressure
plt.plot(df["Elapsed Time (ms)"], df["SystolicBP"])
plt.plot(df["Elapsed Time (ms)"], df["DiastolicBP"])
plt.title("Blood Pressure Over Time")
plt.xlabel("Time (ms)")
plt.ylabel("BP (mmHg)")
plt.legend(["Systolic", "Diastolic"])
plt.show()

# Blood Oxygen Saturation
plt.plot(df["Elapsed Time (ms)"], df["SpO2"])
plt.title("Blood Oxygen Saturation Over Time")
plt.xlabel("Time (ms)")
plt.ylabel("SpO2 (%)")
plt.show()

# Prepare data
features = torch.tensor(df.drop("SpO2", axis=1).values)
target = torch.tensor(df["SpO2"].values)
dataset = TensorDataset(features, target)
dataloader = DataLoader(dataset, batch_size=64)

# Model architecture
model = nn.Sequential(
    nn.Linear(8, 128),  # Updated input size to match the number of features
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
)


# Training
num_epochs = 100
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(num_epochs):
    for X, y in dataloader:
        X = X.float()  # Convert input tensor to float32
        y = y.float()
        pred = model(X)
        loss = criterion(pred, y.view(-1, 1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Print final loss
print("Final Loss:", loss.item())

print("Finished training")
