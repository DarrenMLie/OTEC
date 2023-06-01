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
import math
from thermoModel import *

import CoolProp as CP
from CoolProp.HumidAirProp import HAPropsSI
from CoolProp.CoolProp import PropsSI
import seaborn
import scipy as sp
from scipy.integrate import quad
from global_land_mask import globe
import os

# setup global variables
minLat = -90    # minimum latitude
maxLat = 90     
minLong = 180
maxLong = -180
minDep = 0
maxDep = 10000
startDate = 0
endDate = 10000
areachunk = 1
depthchunk = 10
datechunk = 30

depRange = np.arange(minDep, maxDep, depthchunk)
latRange = np.arange(minLat, maxLat, areachunk)
longRange = np.arange(minLong, maxLong, areachunk)
dateRange = np.arange(startDate, endDate, datechunk)

# Potential Inputs:​ Lat, Long,​ Depths​ & Land Mask, Bathymetric Mask​
def generatePoints():
    point_df = pd.DataFrame(columns = ['Latitude', 'Longitude', 'Depth', 'JulianDate'])
    for date in dateRange:
        for lat in latRange:
            for long in longRange:
                for dep in depRange:
                    # Land Mask
                    if globe.is_land(lat, long):
                        continue
                    # Bathymetric Mask​
                    # if is_too_deep(lat, long):
                    #     continue
                    new_point = {'Latitude':lat, 'Longitude':long, 'Depth':dep, 'JulianDate':date}
                    point_df = point_df.append(new_point, ignore_index=True)
    return point_df

# ML Model​
def getModelResults(point_df):
    pass

# Process predicted thermoclines for Thermo inputs​
def buildThermocline(lat, long, temp_df):
    tc_data = temp_df
    tc_data = tc_data[tc_data['Latitude'].between(lat, lat+areachunk)]
    tc_data = tc_data[tc_data['Longitude'].between(long, long+areachunk)]
    tc_temp = tc_data['Temperature']
    tc_depth = tc_data['Depth']

    plt.plot(tc_temp, tc_depth)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.xlabel("Temperature (degC)")
    plt.ylabel("Depth (m)")
    plt.title(f"Thermocline for Latitude: {lat}-{lat+areachunk} Longitude: {long}-{long+areachunk}", fontsize= 15, fontweight='bold')
    plt.savefig(f"Thermocline_Lat_{lat}_to_{lat+areachunk}_Long_{long}_to_{long+areachunk}.png")
    plt.show()

# Thermodynamic Model​ functions
def getExergyResults():
    #conditions at the top of the ocean
    T_s=300                              #surface water temperature [K]
    P_s=101325                           #surface/atmospheric pressure [Pa]
    D_s=50                               #approximate depth of the surface water (we need to determine a criteria for this)
    T_s_b=290                            #assumed temperature at which the surface zone ends [K], or temperature at D_s
    
    #conditions at the bottom of the ocean (or wherever cold water is drawn from)
    T_b=275                              #temperature of water at the "bottom" [K]

    ex_upper, ex,  ex_lower = errPlotting(T_s,D_s,T_s_b)

    return ex_upper, ex,  ex_lower

# create the exergy map databse
def buildExergyMap():
    exergy_df = pd.DataFrame(columns = ['Latitude', 'Longitude', 'JulianDate', 'Exergy'])
    for date in dateRange:
        for lat in latRange:
            for long in longRange:
                    new_point = {'Latitude': lat, 'Longitude': long, 'JulianDate': date}
                    exergy_df = exergy_df.append(new_point, ignore_index=True)
    return exergy_df
# Result Plots​
def generatePlots():
    # finding specific exergy, normalized exergy availability, and grid exergy availability
    rho=1000                                #density of liquid water
    A_grid=100                              #grid area in m^2
    ex, err = quad(specEx, 0, D_s)
    Ex_norm=ex_int*rho
    Ex_grid=Ex_norm*A_grid
    # generate exergy plots

def main():
    resultsPath = "Results"
    thermoclinePath = resultsPath + "/ThermoclinePlots"
    exergyPath = resultsPath + "/ExergyPlots"
    # create results path if it doesn't exist
    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)

    # dataframe with depth column
    point_df = generatePoints()
    temp_df = getModelResults(point_df)

    # dataframe without depth column for exergy mapping
    exergy_arr = []
    exergy_df = buildExergyMap(exergy_arr)
    for lat in latRange:
        for long in longRange:
            # build thermocline and get exergy values for a specific location on the globe
            temp_arr = buildThermocline(lat, long, temp_df, thermoclinePath)
            ex_upper, ex,  ex_lower = getExergyResults(temp_arr, exergyPath)
            exergy_arr.append(ex)
    
    exergy_df['Exergy'] = exergy_arr
    
    # create ex_map base on selected date
    date = 0
    generatePlots(exergy_df, date)
