#!/usr/bin/env python
# coding: utf-8

# Data Information
train_dataset = 'practiceData3Years.nc'
response_var = 'Temperature'
sensor_var = 'Probe'
# Model Information
num_samples = 200000
num_epochs = 500
batch_size = 64
learning_rate = 0.001
L2 = 0.0001
EarlyStopperPatience = 5
EarlyStopperDelta = 0

# Libraries
import xarray
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda
from torch.utils.data import Dataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
print('how many GPUs? = ',torch.cuda.device_count())

# Load Training Data - here DynaMO chirp-step cycle data
ds = xarray.open_dataset(train_dataset)
# ds = nc.Dataset(train_dataset)
df_raw = ds.to_dataframe()
# df_raw = pd.read_csv(train_dataset, header=0)
num_rows = df_raw.shape[0]
# num_samples = num_rows
# Randomly sample from data set
df = df_raw.sample(num_samples, random_state=1)
# print(df.head())
# df = df.drop(sensor_var, axis=1)
# log transform
df[response_var] = np.log(df[response_var])
# Normalize and clean data
df_mu = df.mean(axis=0)
df_std = df.std()
df_minusmean = df - df_mu
df_normalized = df_minusmean / df_std
df_dropped = df_normalized.dropna()
df_response = df_dropped.pop(response_var)

# Randomly split into training and test data features and response (80% training / 10% validation / 10% test)
X = np.array(df_dropped)
Y = np.array(df_response)
input_size = X.shape[1]
train_input, valid_test_input, train_output, valid_test_output = train_test_split(X, Y, test_size=0.2)
valid_input, test_input, valid_output, test_output = train_test_split(valid_test_input, valid_test_output, test_size=0.5)

# Convert to tensors
train_x = torch.from_numpy(np.float32(train_input))
train_y = torch.from_numpy(np.float32(train_output))
valid_x = torch.from_numpy(np.float32(valid_input))
valid_y = torch.from_numpy(np.float32(valid_output))
test_x = torch.from_numpy(np.float32(test_input))
test_y = torch.from_numpy(np.float32(test_output))
train_y = train_y.unsqueeze(1)
valid_y = valid_y.unsqueeze(1)
test_y = test_y.unsqueeze(1)
# Send tensors to GPU
# train_x = train_x.to(device)
# train_y = train_y.to(device)
valid_x = valid_x.to(device)
valid_y = valid_y.to(device)
test_x = test_x.to(device)
test_y = test_y.to(device)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 100)
        self.fc6 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x


net = Net()
net.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=L2)
print(net)

# Early Stopping Function
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
early_stopper = EarlyStopper(patience=EarlyStopperPatience, min_delta=EarlyStopperDelta)


min_valid_loss = np.inf
tloss = []
vloss = []
# Training Loop
print("Training Loop")
for epoch in range(num_epochs):

	perm = torch.randperm(train_x.size()[0])

	for i in range(0, train_x.size()[0], batch_size):
		
		# Training
		train_loss = 0.0
		net.train()
		optimizer.zero_grad()
		# Training Batch
		train_indices = perm[i:i+batch_size]
		tbatch_x, tbatch_y = train_x[train_indices], train_y[train_indices]
		tbatch_x = tbatch_x.to(device)
		tbatch_y = tbatch_y.to(device)
		# Train step
		Y_predict = net(tbatch_x)
		loss = criterion(Y_predict, tbatch_y)
		loss.backward()
		optimizer.step()
		train_loss += loss.item()
	
	tloss.append(train_loss)	
	# Validation
	valid_loss = 0.0
	net.eval()
	# Valid Evaulation
	V_predict = net(valid_x)
	loss = criterion(V_predict, valid_y)
	valid_loss += loss.item()
	vloss.append(valid_loss)

	print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_input)} \t\t Validation Loss: {valid_loss / len(valid_input)}')
	# Early Stopping
	if early_stopper.early_stop(valid_loss):
		print('Training finished due to Early Stopping')
		break

# Save Model	
torch.save(net.state_dict(), 'OTEC_miniBatchState.pth')
torch.save(net,'OTEC_miniBatch.pth')

# Test Model
test_predictions = net(test_x).detach()
test_predictions.cpu()
# UnNormalizing and undoing log transform
test_predictions_unstandardized = test_predictions * df_std[response_var]
test_predictions_remean = test_predictions_unstandardized + df_mu[response_var]
test_predictions_restored = np.exp(test_predictions_remean.cpu())

test_labels_unstandardized = test_output * df_std[response_var]
test_labels_remean = test_labels_unstandardized + df_mu[response_var]
test_labels_restored = np.exp(test_labels_remean)

# Error Metric Calculation
rms = mean_squared_error(test_labels_restored, test_predictions_restored)
nrms = rms / (df_mu[response_var])
R2 = r2_score(test_labels_restored, test_predictions_restored)
print("Test Data Error Metrics")
print("RMSE = {:.4f}".format(rms))
print("NRMSE = {:.4f}".format(nrms))
print("R2 = {:.4f}".format(R2))

# Loss Plot
epoch_list = range(epoch+1)
fig, ax = plt.subplots(dpi = 150)
ax.plot(epoch_list, tloss, 'k-')
ax.plot(epoch_list, vloss, 'b-')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend(['Training Loss', 'Validation Loss'])
ax.set_title('Loss Plot')
plt.savefig('Loss Plot.png')

xy = np.vstack([test_labels_restored, test_predictions_restored])
z = gaussian_kde(xy)(xy)

# Parity Plot
fig, ax = plt.subplots(dpi = 100)
ax.plot(test_labels_restored, test_predictions_restored, c=z, s=10)
ax.plot(test_labels_restored,test_labels_restored,'r-')
ax.set_xlabel('Measured Temperature')
ax.set_ylabel('Prediction Temperature')
ax.set_title('Parity Plot on Three Year data')
plt.savefig('Parity_3_Year_Data.png')


