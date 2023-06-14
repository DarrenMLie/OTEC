# OTEC Pipeline
# Potential Inputs:​ Lat, Long,​ Depths​
# Land Mask and Bathymetric Mask​
# ML Model​
# Process predicted thermoclines for Thermo inputs​
# Thermodynamic Model​
# Result Plots​

#Import some libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from thermoModel import *
from generateThermocline import generateThermocline
from scipy.integrate import quad
from global_land_mask import globe
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
from julianToNormal import jd_to_date

import os

# START HERE ** do 5 deg chunk of area on weekly basis first, close plot figures to save memory, what is depth units?
# TODO: Test Code with 1 date
# setup global variables
minLat = -90    # minimum latitude
maxLat = 90     
minLong = -180
maxLong = 180
# minDep = 0
maxDep = 1000
startDate = 2455562.5
endDate = 2455927.5
areaIncr = 10    # 5 deg
depthIncr = 5   # 1 m?
dateIncr = 7    # weekly

savePlots = True

# depRange = np.arange(minDep, maxDep, depthIncr)
latRange = np.arange(minLat, maxLat, areaIncr)
longRange = np.arange(minLong, maxLong, areaIncr)
dateRange = np.arange(startDate, endDate, dateIncr)

class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(4, 100)
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

def main():
    resultsPath = "Results"
    thermoclinePath = resultsPath + "/ThermoclinePlots"
    exergyPath = resultsPath + "/ExergyPlots"
    # create results path if it doesn't exist
    print("Creating Directories")
    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)
        os.makedirs(thermoclinePath)
        os.makedirs(exergyPath)
    print("Directories Created Successfully")

    modelPath = "OTEC_miniBatchState.pth"
    net = Net()
    net.load_state_dict(torch.load(modelPath))
    net.eval()

    print("Generating Thermoclines") 
    for date in dateRange:
        for lat in latRange:
            for long in longRange:
                # Land Mask
                if not globe.is_land(lat, long):
                    print(lat, long)
                    
                    # Bathymetric Mask​
                    # if is_too_deep(lat, long):
                    #     continue
                    generateThermocline(lat, long, date, maxDep, depthIncr, net, thermoclinePath, savePlots)
    # generateThermocline(0,0,2455562.5,1000,1,net,thermoclinePath)
if __name__ == "__main__":
    main()