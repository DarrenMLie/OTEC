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

# TODO ** create world plots, validate exergy results with warsinger (order of 10^19), what is depth units?
# NOTE: Everything is validated except for exergy calculations, depth units, and comparison with actual data
# setup global variables
minLat = -90        # minimum latitude (degrees)
maxLat = 90         # maximum latitude (degrees)
minLong = -180      # minimum longitude (degrees)
maxLong = 180       # maximum longitude (degrees)
# minDep = 0
# maxDep = 1000     # determined automatically
startDate = 2455562.5   # start date (julian time)
endDate = 2455927.5     # end date (julian time)
areaIncr = 20       # world area grid (degrees)
depthIncr = 5       # depth increment (meter)
dateIncr = 300        # date increment (days)
tempCutoff = 2      # thermocline temperature cutoff (K or degC)

saveThermoPlots = False
saveExcelData = True

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
    resultsCSVPath = resultsPath + "/ResultsCSV"
    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)
        os.makedirs(thermoclinePath)
        os.makedirs(exergyPath)
        os.makedirs(resultsCSVPath)
        print("Directories Created Successfully")

    # import ML model
    print("Importing model....")
    modelPath = "OTEC_miniBatchState.pth"
    net = Net()
    net.load_state_dict(torch.load(modelPath))
    net.eval()

    # main for loop to generate thermocline and get exergy for each time
    print("Entering main loop....") 
    for date in dateRange:
        # generate new dataframe for storing plotting variables worldwide for each new date
        print(f"Initializing plots dataframe for {date:.1f}....")
        plotDf = pd.DataFrame(columns = ['Latitude', 'Longitude', 'Exergy', 'Surface_Temp', 'Thermocline_Depth'])
    
        # generate data for each location 
        print(f"Generating data for {date:.1f}....")
        for lat in latRange:
            for long in longRange:
                # land mask
                if not globe.is_land(lat, long):
                    # get maximum depth of location from database
                    maxDep = maxDepth(df, lat, long, areaIncr) 

                    # generate thermocline for location using ML model
                    print(f"Generating thermoclines for {lat:.1f}_{long:.1f}_{date:.1f}....") 
                    tempDf = generateThermocline(lat, long, date, maxDep, depthIncr, net, thermoclinePath, saveThermoPlots)
                    
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

        # generate plots (under construction) ** START HERE
        # print(f"Generating world plots for {date:.1f}....")             
        # generatePlots(plotDf)

        # border
        print(f"------------------------------------------------------------------------------------------")
                    
if __name__ == "__main__":
    main()

# hand calculations to verify code calculations
# validate predictions by comparing to actual data
# validate thermo model using other paper resource
# global exergy map, global thermocline depth map, global map of surface temp
# depth vs location exergy contour, locaiton vs time vs exergy contour