import numpy as np
import CoolProp as CP
import matplotlib.pyplot as plt
from CoolProp.HumidAirProp import HAPropsSI
from CoolProp.CoolProp import PropsSI
import pandas as pd
from generateThermocline import generateThermocline
import torch

# 1 degree = 111139 m
def degreeToMeters(deg):
    return deg*111139

# find dead state of thermocline
def findDeadStateIndex(tempDf, tempCutoff):
    TLast = tempDf.min()["Temperature"]
    for i in range(len(tempDf)):
        if tempDf.loc[i, "Temperature"] < (TLast + tempCutoff):
            break

    return i

# calculates exergy for a chunk of water in the ocean from depth 1 to depth 2
def calculateSingleExergy(depthIncr, TTop, TDead, areaIncrMeter):
    P0 = 101325 # Pa
    rho = 1000     # kg/m^3
    hTop = PropsSI('H', 'T', TTop+273.153, 'P', P0, 'Water')      #enthalpy of surface water [J/kg]
    hDead = PropsSI('H', 'T', TDead+273.153, 'P', P0, 'Water')       #minimum possible enthalpy of the surface water [J/kg]
    sTop = PropsSI('S', 'T', TTop+273.153, 'P', P0, 'Water')      #entropy of surface water [J/kg-K]
    sDead = PropsSI('S', 'T', TDead+273.153, 'P', P0, 'Water')       #minimum possible entropy of the surface water [J/kg-K]

    specificExergy = (hTop - hDead) - (TDead+273.153)*(sTop - sDead) 
    exergy = specificExergy * depthIncr * rho * areaIncrMeter**2

    return exergy

# calculates total exergy for current location and returns total exergy + thermocline depth/dead state depth based on criteria
def calculateTotalExergy(tempDf, depthIncr, areaIncr, tempCutoff):
    totalExergy = 0
    areaIncrMeter = degreeToMeters(areaIncr)
    deadStateIndex = findDeadStateIndex(tempDf, tempCutoff)
    for i in range(deadStateIndex): # not including dead state because dead state to itself = 0 exergy
        TTop = tempDf.loc[i, "Temperature"]
        TDead = tempDf.loc[deadStateIndex, "Temperature"]
    
        exergy = calculateSingleExergy(depthIncr, TTop, TDead, areaIncrMeter)
        totalExergy += exergy

    return totalExergy, deadStateIndex

if __name__ == "__main__":
    from Net import Net
    modelPath = "OTEC_miniBatchState.pth"
    net = Net()
    net.load_state_dict(torch.load(modelPath))
    net.eval()
    tempDf = generateThermocline(0,0,2455562.5,2000,1,net,"Results/ThermoclinePlots")
    print(tempDf)
    print(calculateTotalExergy(tempDf, 1, 5, 3))
    # totalExergy = calculateTotalExergy(tempDf, 1, 5, 2)