import numpy as np
import CoolProp as CP
import matplotlib.pyplot as plt
from CoolProp.HumidAirProp import HAPropsSI
from CoolProp.CoolProp import PropsSI
import pandas as pd
import seaborn
import scipy as sp

#conditions at the top of the ocean
T_s=300                              #surface water temperature [K]
P_s=101325                           #surface/atmospheric pressure [Pa]
D_s=50                               #approximate depth of the surface water (we need to determine a criteria for this)
T_s_b=290                            #assumed temperature at which the surface zone ends [K], or temperature at D_s

#conditions at the bottom of the ocean (or wherever cold water is drawn from)
T_b=275                              #temperature of water at the "bottom" [K]


def T(y,T_s,T_s_b,D_s):
  return T_s-(T_s-T_s_b)/D_s*y

def specEx(y):
  h_s = PropsSI('H', 'T', T(y,T_s,T_s_b,D_s), 'P', P_s, 'Water')      #enthalpy of surface water [J/kg]
  h_s_c=PropsSI('H', 'T', T_b, 'P', P_s, 'Water')       #minimum possible enthalpy of the surface water [J/kg]
  s_s = PropsSI('S', 'T', T(y,T_s,T_s_b,D_s), 'P', P_s, 'Water')      #entropy of surface water [J/kg-K]
  s_s_c=PropsSI('S', 'T', T_b, 'P', P_s, 'Water')       #minimum possible entropy of the surface water [J/kg-K]

  ex=h_s-h_s_c-T_b*(s_s-s_s_c)                          #exergy calculation

  return ex/1000

from scipy.integrate import quad
rho=1000                                #density of liquid water
A_grid=100                              #grid area in m^2
ex_int, err = quad(specEx, 0, D_s)
Ex_norm=ex_int*rho
Ex_grid=Ex_norm*A_grid

print(ex_int)
print(Ex_norm)
print(Ex_grid)

def specExUpper(y):
    error=0.1
    h_s = PropsSI('H', 'T', (1+error)*T(y,T_s,T_s_b,D_s), 'P', P_s, 'Water')      #enthalpy of surface water [J/kg]
    h_s_c=PropsSI('H', 'T', T_b, 'P', P_s, 'Water')       #minimum possible enthalpy of the surface water [J/kg]
    s_s = PropsSI('S', 'T', (1+error)*T(y,T_s,T_s_b,D_s), 'P', P_s, 'Water')      #entropy of surface water [J/kg-K]
    s_s_c=PropsSI('S', 'T', T_b, 'P', P_s, 'Water')       #minimum possible entropy of the surface water [J/kg-K]

    ex=h_s-h_s_c-T_b*(s_s-s_s_c)                          #exergy calculation

    return ex/1000

def specExLower(y):
    error=0.1
    h_s = PropsSI('H', 'T', (1-error)*T(y,T_s,T_s_b,D_s), 'P', P_s, 'Water')      #enthalpy of surface water [J/kg]
    h_s_c=PropsSI('H', 'T', T_b, 'P', P_s, 'Water')       #minimum possible enthalpy of the surface water [J/kg]
    s_s = PropsSI('S', 'T', (1-error)*T(y,T_s,T_s_b,D_s), 'P', P_s, 'Water')      #entropy of surface water [J/kg-K]
    s_s_c=PropsSI('S', 'T', T_b, 'P', P_s, 'Water')       #minimum possible entropy of the surface water [J/kg-K]

    ex=h_s-h_s_c-T_b*(s_s-s_s_c)                          #exergy calculation

    return ex/1000

def errPlotting(T_s,D_s,T_s_b):

    ex_upper, e_upper=quad(specExUpper, 0, D_s)
    ex, err=quad(specEx,0,D_s)
    ex_lower, e_lower=quad(specExLower, 0, D_s)
    

    return ex_upper, ex,  ex_lower

errPlotting(310,50,275)