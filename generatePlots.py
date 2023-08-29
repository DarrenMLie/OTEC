#!/usr/bin/env python
# coding: utf-8
"""
generatePlots.py

This file contains functions that create world maps and plots of the various recorded metrics obtained from the 
simulation results such as surface temperature, exergy, ocean depth, thermocline depth, etc.

@author Darren Lie
@version August 28, 2023

"""

# Import Libraries
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np
import geodatasets
from julianToNormal import jd_to_date

# marker options
# - s=0.1, linewidths=0 (too small)
# - marker = "," (too big)
# - marker = "." (best option right now)

# world map plotting settings
plotColor = "lightgrey"
opacity = 0.3

# function that creates the global exergy map
def generateExergyMap(plotDf, exergyPath, date):
    worldmap = gpd.read_file(geodatasets.get_path("naturalearth.land"))

    # Creating axes and plotting world map
    fig, ax = plt.subplots(figsize=(12, 6))
    worldmap.plot(color=plotColor, ax=ax)

    # Plotting our Impact Energy data with a color map
    x = plotDf['Longitude']
    y = plotDf['Latitude']
    z = plotDf['Exergy']
    plt.scatter(x, y, c=z, alpha=opacity, marker=".")
    plt.colorbar(label='Exergy (J/m$^2$)')
    plt.title("Global Thermocline Exergy Map")
    plt.xlabel("Longitude ($^\circ$)")
    plt.ylabel("Latitude ($^\circ$)")

    year,month,day = jd_to_date(date)
    plt.savefig(f'{exergyPath}/Exergy_{int(month)}-{int(day)}-{str(int(year))[2:]}.png')
    plt.close('all')

# function that creates the global surface temperature map
def generateSurfaceTempMap(plotDf, surfaceTempPath, date):
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
    plt.savefig(f'{surfaceTempPath}/SurfaceTemp_{int(month)}-{int(day)}-{str(int(year))[2:]}.png')
    plt.close('all')

# function that creates the global thermocline depth map
def generateThermoDepthMap(plotDf, thermoDepthPath, date):
    worldmap = gpd.read_file(geodatasets.get_path("naturalearth.land"))

    # Creating axes and plotting world map
    fig, ax = plt.subplots(figsize=(12, 6))
    worldmap.plot(color=plotColor, ax=ax)

    # Plotting our Impact Energy data with a color map
    x = plotDf['Longitude']
    y = plotDf['Latitude']
    z = plotDf['Thermocline_Depth']
    plt.scatter(x, y, c=z, alpha=opacity, marker=".")
    plt.colorbar(label='Thermocline Depth (m)')
    plt.title("Global Thermocline Depth Map")
    plt.xlabel("Longitude ($^\circ$)")
    plt.ylabel("Latitude ($^\circ$)")

    year,month,day = jd_to_date(date)
    plt.savefig(f'{thermoDepthPath}/ThermoDepth_{int(month)}-{int(day)}-{str(int(year))[2:]}.png')
    plt.close('all')

# function that creates the global ocean depth map
def generateOceanDepthMap(plotDf, oceanDepthPath, date):
    worldmap = gpd.read_file(geodatasets.get_path("naturalearth.land"))

    # Creating axes and plotting world map
    fig, ax = plt.subplots(figsize=(12, 6))
    worldmap.plot(color=plotColor, ax=ax)

    # Plotting our Impact Energy data with a color map
    x = plotDf['Longitude']
    y = plotDf['Latitude']
    z = plotDf['Ocean_Depth']
    plt.scatter(x, y, c=z, alpha=opacity, marker=".")
    plt.colorbar(label='Ocean Depth (m)')
    plt.title("Global Ocean Depth Map")
    plt.xlabel("Longitude ($^\circ$)")
    plt.ylabel("Latitude ($^\circ$)")

    year,month,day = jd_to_date(date)
    plt.savefig(f'{oceanDepthPath}/OceanDepth_{int(month)}-{int(day)}-{str(int(year))[2:]}.png')
    plt.close('all')

# function that calls all other plotting functions
def generatePlots(plotDf, date, exergyPath, surfaceTempPath, thermoDepthPath, oceanDepthPath):
    generateExergyMap(plotDf, exergyPath, date)
    generateSurfaceTempMap(plotDf, surfaceTempPath, date)
    generateThermoDepthMap(plotDf, thermoDepthPath, date)
    generateOceanDepthMap(plotDf, oceanDepthPath, date)

# function for testing plotting methods
def geopandasTest(plotDf):
    # # GEOPANDAS DOCS
    # path = get_path("naturalearth.land")
    # worldmap = gpd.read_file(path)
    # ax = worldmap.plot()
    # plt.show()

    # EASY WAY TO PLOT DOCS (works simply)
    # From GeoPandas, our world map data
    # worldmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    # From geodatasets
    worldmap = gpd.read_file(geodatasets.get_path("naturalearth.land"))

    # Creating axes and plotting world map
    fig, ax = plt.subplots(figsize=(12, 6))
    worldmap.plot(color="lightgrey", ax=ax)

    # Plotting our Impact Energy data with a color map
    x = plotDf['Longitude']
    y = plotDf['Latitude']
    z = plotDf['Exergy']
    # plt.scatter(x, y, s=20*z, c=z, alpha=0.6, vmin=0, vmax=threshold, cmap='autumn')
    plt.scatter(x, y, c=z, alpha=0.6)
    plt.colorbar(label='Exergy (J)')

    # Creating axis limits and title
    # plt.xlim([-180, 180])
    # plt.ylim([-90, 90])

    plt.title("Exergy Map")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
    
if __name__ == "__main__":
    plotDf = pd.read_csv("Results/ResultsCSV/Results_1-1-11.csv")
    resultsPath = "Results"
    exergyPath = resultsPath + "/ExergyMaps"
    surfaceTempPath = resultsPath + "/SurfaceTempMaps"
    thermoDepthPath = resultsPath + "/ThermoclineMaps"
    oceanDepthPath = resultsPath + "/OceanDepthMaps"
    resultsCSVPath = resultsPath + "/ResultsCSV"
    
    generatePlots(plotDf, 2455562.5, exergyPath, surfaceTempPath, thermoDepthPath, oceanDepthPath)
    # geopandasTest(plotDf)