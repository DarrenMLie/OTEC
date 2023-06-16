# OTEC Pipeline
# Potential Inputs:​ Lat, Long,​ Depths​
# Land Mask and Bathymetric Mask​
# ML Model​
# Process predicted thermoclines for Thermo inputs​
# Thermodynamic Model​
# Result Plots​

#Import some libraries
import numpy as np
# from thermoModel import *
from generateThermocline import generateThermocline
from maxDepth import maxDepth
from Net import Net
from global_land_mask import globe
import os
import xarray
import torch

# START HERE ** delete pictures, start making thermo model, what is depth units?
# setup global variables
minLat = -90        # minimum latitude
maxLat = 90     
minLong = -180
maxLong = 180
# minDep = 0
# maxDep = 1000     # determined automatically
startDate = 2455562.5
endDate = 2455927.5
areaIncr = 20       # 5 deg
depthIncr = 5       # 1 m?
dateIncr = 7        # weekly

savePlots = True

latRange = np.arange(minLat, maxLat, areaIncr)
longRange = np.arange(minLong, maxLong, areaIncr)
dateRange = np.arange(startDate, endDate, dateIncr)

def main():
    # import training dataset for depth checking
    print("Importing training data....")
    train_dataset = "practiceData3Years.nc"
    ds = xarray.open_dataset(train_dataset)
    df = ds.to_dataframe()

    # create results directories if it doesn't exist
    print("Creating directories....")
    resultsPath = "Results"
    thermoclinePath = resultsPath + "/ThermoclinePlots"
    exergyPath = resultsPath + "/ExergyPlots"
    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)
        os.makedirs(thermoclinePath)
        os.makedirs(exergyPath)
        print("Directories Created Successfully")

    # import ML model
    print("Importing model....")
    modelPath = "OTEC_miniBatchState.pth"
    net = Net()
    net.load_state_dict(torch.load(modelPath))
    net.eval()

    # Generate thermoclines for all points
    print("Generating thermoclines....") 
    for date in dateRange:
        for lat in latRange:
            for long in longRange:
                # Land Mask
                if not globe.is_land(lat, long):
                    print(lat, long)
                    maxDep = maxDepth(df, lat, long, areaIncr) # get maximum depth of point from database
                    generateThermocline(lat, long, date, maxDep, depthIncr, net, thermoclinePath, savePlots)
    
    # generateThermocline(0,0,2455562.5,1000,1,net,thermoclinePath)
if __name__ == "__main__":
    main()