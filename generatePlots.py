import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np
from geodatasets import get_path
# from cartopy import crs as ccrs

# START HERE ** find best way to plot contour on world map, geopandas seems kind of sketch
def generateExergyMap(exergyDf):
    # exergyDf.plot(x="Longitude", y="Latitude", kind="scatter", c="blue", colormap="YlOrRd")
    exergyDf.plot(x="Longitude", y="Latitude")

def generateSurfaceTempMap(surfaceTempDf):
    pass

def generateThermoDepthMap(thermoDepthDf):
    pass

def generatePlots(plotDf):
    generateExergyMap(plotDf[["Latitude", "Longitude", "Exergy"]])
    generateSurfaceTempMap(plotDf[["Latitude", "Longitude", "Surface_Temp"]])
    generateThermoDepthMap(plotDf[["Latitude", "Longitude", "Thermocline_Depth"]])

def geopandasTest(plotDf):
    # GEOPANDAS DOCS
    path = get_path("naturalearth.land")
    worldmap = gpd.read_file(path)
    ax = worldmap.plot()
    plt.show()

    # # EASY WAY TO PLOT DOCS (currently not working)
    # # From GeoPandas, our world map data
    # worldmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

    # # Creating axes and plotting world map
    # fig, ax = plt.subplots(figsize=(12, 6))
    # worldmap.plot(color="lightgrey", ax=ax)

    # # Plotting our Impact Energy data with a color map
    # x = plotDf['Longitude']
    # y = plotDf['Latitude']
    # z = plotDf['Exergy']
    # # plt.scatter(x, y, s=20*z, c=z, alpha=0.6, vmin=0, vmax=threshold, cmap='autumn')
    # plt.scatter(x, y, s=20*z, c=z, alpha=0.6, cmap='autumn')
    # plt.colorbar(label='Exergy (J)')

    # # Creating axis limits and title
    # plt.xlim([-180, 180])
    # plt.ylim([-90, 90])

    # plt.title("Exergy Map")
    # plt.xlabel("Longitude")
    # plt.ylabel("Latitude")
    # plt.show()
    
if __name__ == "__main__":
    plotDf = pd.read_csv("Results/ResultsCSV/Results_1-1-11.csv")
    # generatePlots(plotDf)
    geopandasTest(plotDf)