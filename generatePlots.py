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

plotColor = "lightgrey"
opacity = 0.5

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
    plt.colorbar(label='Exergy (J)')
    plt.title("Global Thermocline Exergy Map")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    year,month,day = jd_to_date(date)
    plt.savefig(f'{exergyPath}/Exergy_{int(month)}-{int(day)}-{str(int(year))[2:]}.png')
    plt.close('all')

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
    plt.colorbar(label='Surface Temp (C)')
    plt.title("Global Surface Temperature Map")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    year,month,day = jd_to_date(date)
    plt.savefig(f'{surfaceTempPath}/SurfaceTemp_{int(month)}-{int(day)}-{str(int(year))[2:]}.png')
    plt.close('all')

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
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    year,month,day = jd_to_date(date)
    plt.savefig(f'{thermoDepthPath}/ThermoDepth_{int(month)}-{int(day)}-{str(int(year))[2:]}.png')
    plt.close('all')

def generatePlots(plotDf, date, exergyPath, surfaceTempPath, thermoDepthPath):
    generateExergyMap(plotDf, exergyPath, date)
    generateSurfaceTempMap(plotDf, surfaceTempPath, date)
    generateThermoDepthMap(plotDf, thermoDepthPath, date)

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
    resultsCSVPath = resultsPath + "/ResultsCSV"
    
    generatePlots(plotDf, 2455562.5, exergyPath, surfaceTempPath, thermoDepthPath)
    # geopandasTest(plotDf)