#!/usr/bin/env python
# coding: utf-8

#Import some libraries
import numpy as np
import pandas as pd
# from thermoModel import *
from generateThermocline import generateThermocline
from maxDepth import maxDepth
from julianToNormal import jd_to_date
import matplotlib.pyplot as plt
from Net import Net
from global_land_mask import globe
import os
import xarray
import torch
import geopandas as gpd
import geodatasets

# setup global variables
latRange = [-90, 90]                # range of world latitudes (degrees)
longRange = [-180, 180]             # range of world longitudes (degrees)
dateRange = [2455562.5, 2455927.5]  # range of dates (julian time/days)

areaIncr = 20        # world area grid (degrees)
depthIncr = 5       # depth increment (meter)
dateIncr = 300        # date increment (days)
tempCutoff = 1      # thermocline temperature cutoff (K or degC)

latArray = np.arange(latRange[0], latRange[1], areaIncr)
longArray = np.arange(longRange[0], longRange[1], areaIncr)
dateArray = np.arange(dateRange[0], dateRange[1], dateIncr)

def generateThermoclineActual(df, lat, long, date, thermoActualPath):
    err = 0.1
    dateErr = 1
    filtered_df = df.loc[(df['Latitude'].between(lat - err, lat + err)) 
           & (df['Longitude'].between(long - err, long + err))
           & (df['Julian_Time'].between(date - dateErr, date + dateErr)),['Depth', 'Temperature']]
    sorted_df = filtered_df.sort_values(by=['Depth'])
    dropped_df = sorted_df.drop(columns=['Latitude', 'Longitude', 'Julian_Time'])
    # plot **
    year,month,day=jd_to_date(date)

    fig, ax = plt.subplots()
    ax.plot(dropped_df["Temperature"], dropped_df["Depth"])  
    ax.set_xlabel('Temperature ($^\circ$C)')
    ax.set_ylabel('Depth (m)')
    ax.set_title(f'Thermocline Actual for Lat: {lat:,.1f} Long: {long:,.1f} Date: {int(month)}-{int(day)}-{str(int(year))[2:]}')
    ax.invert_yaxis()
    plt.savefig(f'{thermoActualPath}/ThermoclineActual_{lat:,.1f}_{long:,.1f}_{int(month)}-{int(day)}-{str(int(year))[2:]}.png')
    plt.close('all')

    return dropped_df
    
def generateSurfaceTempActualMap(df, surfaceTempActualPath, date):
    plotDf = df.loc[(df['Julian_Time'].between(date - 1, date + 1))
                    & (df['Depth'] == 0),['Longitude', 'Latitude', 'Temperature']]
    
    plotColor = "lightgrey"
    opacity = 0.3
    worldmap = gpd.read_file(geodatasets.get_path("naturalearth.land"))

    # Creating axes and plotting world map
    fig, ax = plt.subplots(figsize=(12, 6))
    worldmap.plot(color=plotColor, ax=ax)

    # Plotting our Impact Energy data with a color map
    x = plotDf['Longitude']
    y = plotDf['Latitude']
    z = plotDf['Surface_Temp']
    plt.scatter(x, y, c=z, alpha=opacity, marker=".")
    plt.colorbar(label='Surface Temp ($^\circ$C)')
    plt.title("Global Surface Temperature Map")
    plt.xlabel("Longitude ($^\circ$)")
    plt.ylabel("Latitude ($^\circ$)")

    year,month,day = jd_to_date(date)
    plt.savefig(f'{surfaceTempActualPath}/SurfaceTempActual_{int(month)}-{int(day)}-{str(int(year))[2:]}.png')
    plt.close('all')
def modelValidation():
    # import training dataset for depth checking
    print("Importing training data....")
    train_dataset = "practiceData3Years.nc"
    ds = xarray.open_dataset(train_dataset)
    df = ds.to_dataframe()

    # create results directories if it doesn't exist
    print("Creating directories....")
    resultsPath = "ResultsTest"
    thermoPath = resultsPath + "/ThermoclinePlots"
    thermoActualPath = resultsPath + "/ThermoclineActualPlots"
    surfaceTempActualPath = resultsPath + "/SurfaceTempActualMaps"
    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)
        os.makedirs(thermoPath)
        os.makedirs(thermoActualPath)
        os.makedirs(surfaceTempActualPath)
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
                    tempDf = generateThermocline(lat, long, date, maxDep, depthIncr, net, thermoPath, True)
                    tempDfActual = generateThermoclineActual(df, lat, long, date, thermoActualPath)
        
        generateSurfaceTempActualMap(df, surfaceTempActualPath, date)
if __name__ == "__main__":
    modelValidation()
