import pandas as pd
import numpy as np

# START HERE ** WHY ISNT IT PLOTTING ON WORLD MAP
def generateExergyMap(exergyDf):
    exergyDf.plot(x="Longitude", y="Latitude", kind="scatter", c="blue", colormap="YlOrRd")

def generateSurfaceTempMap(surfaceTempDf):
    pass

def generateThermoDepthMap(thermoDepthDf):
    pass

def generatePlots(plotDf):
    generateExergyMap(plotDf[["Latitude", "Longitude", "Exergy"]])
    generateSurfaceTempMap(plotDf[["Latitude", "Longitude", "Surface_Temp"]])
    generateThermoDepthMap(plotDf[["Latitude", "Longitude", "Thermocline_Depth"]])

if __name__ == "__main__":
    plotDf = pd.read_csv("Results/ResultsCSV/Results_1-1-11.csv")
    generatePlots(plotDf)