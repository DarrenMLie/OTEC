# OTEC Main File
# Potential Inputs:​ Lat, Long,​ Depths​
# Land Mask and Bathymetric Mask​
# ML Model​
# Process predicted thermoclines for Thermo inputs​
# Thermodynamic Model​
# Result Plots​

#Import some libraries
import numpy as np
import pandas as pd
# from thermoModel import *
from generateThermocline import generateThermocline
from maxDepth import maxDepth
from calculateExergy import calculateTotalExergy
from generatePlots import generatePlots
from julianToNormal import jd_to_date
from Net import Net
from global_land_mask import globe
import os
import xarray
import torch

# TODO
# - Get better software plotting tool than geopandas or plot in MATLAB
# - Test world plots with smaller incr (~30 mins for 1 points for 5 deg)
# - Validate exergy results with warsinger (order of 10^19)
# - Validate temp results
# - What is depth units?

# NOTE: Everything is validated except for exergy calculations, 
# depth units, and comparison with actual data

# setup global variables
latRange = [-90, 90]                # range of world latitudes (degrees)
longRange = [-180, 180]             # range of world longitudes (degrees)
dateRange = [2455562.5, 2455927.5]  # range of dates (julian time/days)

areaIncr = 5        # world area grid (degrees)
depthIncr = 5       # depth increment (meter)
dateIncr = 7        # date increment (days)
tempCutoff = 2      # thermocline temperature cutoff (K or degC)

saveThermoPlots = False
saveExcelData = True

latArray = np.arange(latRange[0], latRange[1], areaIncr)
longArray = np.arange(longRange[0], longRange[1], areaIncr)
dateArray = np.arange(dateRange[0], dateRange[1], dateIncr)

def main():
    # import training dataset for depth checking
    print("Importing training data....")
    train_dataset = "practiceData3Years.nc"
    ds = xarray.open_dataset(train_dataset)
    df = ds.to_dataframe()

    # create results directories if it doesn't exist
    print("Creating directories....")
    resultsPath = "Results"
    thermoPath = resultsPath + "/ThermoclinePlots"
    exergyPath = resultsPath + "/ExergyMaps"
    surfaceTempPath = resultsPath + "/SurfaceTempMaps"
    thermoDepthPath = resultsPath + "/ThermoclineMaps"
    resultsCSVPath = resultsPath + "/ResultsCSV"
    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)
        os.makedirs(thermoPath)
        os.makedirs(exergyPath)
        os.makedirs(resultsCSVPath)
        os.makedirs(surfaceTempPath)
        os.makedirs(thermoDepthPath)
        print("Directories Created Successfully")

    # import ML model
    print("Importing model....")
    modelPath = "OTEC_miniBatchState.pth"
    net = Net()
    net.load_state_dict(torch.load(modelPath))
    net.eval()

    # main for loop to generate thermocline and get exergy for each time
    print("Entering main loop....") 
    for date in dateArray:
        # generate new dataframe for storing plotting variables worldwide for each new date
        print(f"Initializing plots dataframe for {date:.1f}....")
        plotDf = pd.DataFrame(columns = ['Latitude', 'Longitude', 'Exergy', 'Surface_Temp', 'Thermocline_Depth'])
    
        # generate data for each location 
        print(f"Generating data for {date:.1f}....")
        for lat in latArray:
            for long in longArray:
                # land mask
                if not globe.is_land(lat, long):
                    # get maximum depth of location from database
                    maxDep = maxDepth(df, lat, long, areaIncr) 

                    # generate thermocline for location using ML model
                    print(f"Generating thermoclines for {lat:.1f}_{long:.1f}_{date:.1f}....") 
                    tempDf = generateThermocline(lat, long, date, maxDep, depthIncr, net, thermoPath, saveThermoPlots)
                    
                    # calculate available exergy and thermocline depth for location
                    print(f"Calculating total exergy for {lat:.1f}_{long:.1f}_{date:.1f}....") 
                    totalExergy, thermoDepthIndex = calculateTotalExergy(tempDf, depthIncr, areaIncr, tempCutoff)
                    
                    # add data to plotting dataframe
                    plotDf = plotDf._append({'Latitude' : lat, 
                                            'Longitude' : long, 
                                            'Exergy' : totalExergy,
                                            'Surface_Temp': tempDf.loc[0, "Temperature"], 
                                            'Thermocline_Depth': tempDf.loc[thermoDepthIndex, "Depth"]}, ignore_index = True)

        # save data for current time in excel sheet
        if saveExcelData:
            print(f"Saving data for {date:.1f}....")
            year,month,day = jd_to_date(date)
            plotDf.to_csv(f"{resultsCSVPath}/Results_{int(month)}-{int(day)}-{str(int(year))[2:]}.csv")

        # generate plots
        print(f"Generating world plots for {date:.1f}....")             
        generatePlots(plotDf, date, exergyPath, surfaceTempPath, thermoDepthPath)

        # border
        print(f"------------------------------------------------------------------------------------------")
                    
if __name__ == "__main__":
    main()

# hand calculations to verify code calculations
# validate predictions by comparing to actual data
# validate thermo model using other paper resource
# global exergy map, global thermocline depth map, global map of surface temp
# depth vs location exergy contour, locaiton vs time vs exergy contour