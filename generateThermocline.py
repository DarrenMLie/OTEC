#!/usr/bin/env python
# coding: utf-8
"""
generateThermocline.py

This file contains a function that generates a thermocline for the given latitude and longitude using the trained
neural network to make temperature predictions from the surface down to the maximum depth of the location.

@author Darren Lie
@version August 28, 2023

"""

# Import Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from julianToNormal import jd_to_date
import os

# returns model predictions (as numpy array) for given lat long date and depth limit (plots are optional)
def generateThermocline(lat, long, date, depthLimit, depthIncr, net, folder, savePlots=True):
    # Check whether the specified path exists or not
    isExist = os.path.exists(folder)
    if not isExist:
        os.makedirs(folder)
        print("The new directory is created!")

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    # print(torch.cuda.is_available())
    # print('how many GPUs? = ',torch.cuda.device_count())

    # if there is no data at the location, set everything to 0 to prevent errors
    if depthLimit == 0:
        data = {
            'Latitude': [lat for i in range(0, 1)],
            'Longitude': [long for i in range(0, 1)],
            'Julian_Time': [date for i in range(0, 1)],
            'Depth': [i for i in range(0, 1)]
        }
    else: 
        data = {
            'Latitude': [lat for i in range(0, depthLimit, depthIncr)],
            'Longitude': [long for i in range(0, depthLimit, depthIncr)],
            'Julian_Time': [date for i in range(0, depthLimit, depthIncr)],
            'Depth': [i for i in range(0, depthLimit, depthIncr)]
        }

    df = pd.DataFrame(data)

    # Training Data Mu and Std
    df_mu = pd.read_csv("mu.csv").squeeze()
    df_std = pd.read_csv("std.csv").squeeze()
    temp_train_mu = df_mu.pop("Temperature")
    temp_train_std = df_std.pop("Temperature")

    # Normalize and clean data
    df_normalized = (df - df_mu) / df_std
    df_inputs = df_normalized.dropna()

    # Convert to numpy arrays
    df_inputs_arr = np.array(df_inputs)

    # Convert to tensors
    model_inputs = torch.from_numpy(np.float32(df_inputs_arr))

    # Send tensors to GPU
    model_inputs = model_inputs.to(device)

    # Test Model
    model_output = net(model_inputs).detach()
    model_output.cpu()
    # UnNormalizing and undoing log transform
    model_output_unstandardized = model_output * temp_train_std 
    model_output_remean = model_output_unstandardized + temp_train_mu
    model_output_restored = np.exp(model_output_remean.cpu())

    # Pack temperature results into depth-temp dataframe
    depthArr = data["Depth"]
    tempArr = [item[0] for item in model_output_restored.tolist()]
    tempDf = pd.DataFrame(list(zip(depthArr, tempArr)), columns =["Depth", "Temperature"])

    # save plots if the savePlots flag is set to True
    if savePlots:
        year,month,day=jd_to_date(date)

        fig, ax = plt.subplots()
        ax.plot(model_output_restored, data["Depth"])  
        ax.set_xlabel('Temperature ($^\circ$C)')
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'Thermocline for Lat: {lat:,.1f} Long: {long:,.1f} Date: {int(month)}-{int(day)}-{str(int(year))[2:]}')
        ax.invert_yaxis()
        plt.savefig(f'{folder}/Thermocline_{lat:,.1f}_{long:,.1f}_{int(month)}-{int(day)}-{str(int(year))[2:]}.png')
        plt.close('all')
    return tempDf

if __name__ == "__main__":
    from Net import Net
    modelPath = "OTEC_miniBatchState.pth"
    net = Net()
    net.load_state_dict(torch.load(modelPath))
    net.eval()
    tempDf = generateThermocline(70,-120,2455562.5,1000,1,net,"Results/ThermoclinePlots")
    print(tempDf)
    print(tempDf['Temperature'].iloc[0])

    # generateThermocline(0,0,2455562.5+182,1000,1,net,"Results/ThermoclinePlots")
    # generateThermocline(0,0,2455562.5,1000,1,net,"Results/ThermoclinePlots")
    # generateThermocline(0,0,2455562.5+182,1000,1,net,"Results/ThermoclinePlots")