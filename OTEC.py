#!/usr/bin/env python
# coding: utf-8
"""
OTEC.py

This file is the main script to run the OTEC project simulation to generate thermocline predictions around the globe
and create world maps for various parameters and results.

@author Darren Lie
@version August 28, 2023

"""
# OTEC Main File
# Potential Inputs:​ Lat, Long,​ Depths​
# Land Mask and Bathymetric Mask​
# ML Model​
# Process predicted thermoclines for Thermo inputs​
# Thermodynamic Model​
# Result Plots​

# Import Libraries
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
# - Test out different markers for best size -> "." marker is good
# - Test world plots with smaller incr (~30 mins for 1 points for 5 deg) v -> 1 degree looks good
# - Validate exergy results with warsinger (order of 10^19) -> v makes sense
# - Validate temp results v -> need more data
# - What is depth units? v -> probably m

# - parse 20 year data set and retrain model
# - parallelize sim to speed up process
# - inputs using day of year for seasons? use different AI model?
# - write complete report using new plots and new data
# - additional plots
# - investigate if using contour plots are better than scatter plots 
# - Depth changes each year? Might need to impose constraint on the max depth
# NOTE: Everything is validated except for exergy calculations, 
# depth units, and comparison with actual data

# setup global variables
latRange = [-90, 90]                # range of world latitudes (degrees)
longRange = [-180, 180]             # range of world longitudes (degrees)
dateRange = [2455562.5, 2455927.5]  # range of dates (julian time/days)

areaIncr = 1        # world area grid (degrees)
depthIncr = 1       # depth increment (meter)
dateIncr = 7        # date increment (days)
tempCutoff = 1      # thermocline temperature cutoff (K or degC)

saveThermoPlots = False # True = save thermocline plots (don't recommend if a lot of points are simulated)
saveExcelData = True    # True = save world map data into excel files (recommend this to keep track of data)

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
    oceanDepthPath = resultsPath + "/OceanDepthMaps"
    resultsCSVPath = resultsPath + "/ResultsCSV"
    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)
        os.makedirs(thermoPath)
        os.makedirs(exergyPath)
        os.makedirs(resultsCSVPath)
        os.makedirs(surfaceTempPath)
        os.makedirs(thermoDepthPath)
        os.makedirs(oceanDepthPath)
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
        plotDf = pd.DataFrame(columns = ['Latitude', 'Longitude', 'Exergy', 'Surface_Temp', 'Thermocline_Depth', 'Ocean_Depth'])
    
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
                                            'Surface_Temp': tempDf['Temperature'].iloc[0], 
                                            'Thermocline_Depth': tempDf['Depth'].iloc[thermoDepthIndex], 
                                            'Ocean_Depth': tempDf.iloc[-1]["Depth"]}, ignore_index = True)
                    # Note: tempDf.loc[0, "Depth"] would get index #0 not the 1st element
        # save data for current time in excel sheet
        if saveExcelData:
            print(f"Saving data for {date:.1f}....")
            year,month,day = jd_to_date(date)
            plotDf.to_csv(f"{resultsCSVPath}/Results_{int(month)}-{int(day)}-{str(int(year))[2:]}.csv")

        # generate plots
        print(f"Generating world plots for {date:.1f}....")             
        generatePlots(plotDf, date, exergyPath, surfaceTempPath, thermoDepthPath, oceanDepthPath)

        # border
        print(f"------------------------------------------------------------------------------------------")
                    
if __name__ == "__main__":
    main()

# hand calculations to verify code calculations
# validate predictions by comparing to actual data
# validate thermo model using other paper resource
# global exergy map, global thermocline depth map, global map of surface temp
# depth vs location exergy contour, locaiton vs time vs exergy contour