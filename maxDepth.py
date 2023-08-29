
#!/usr/bin/env python
# coding: utf-8
"""
maxDepth.py

This file contains a function that determines the maximum depth of a certain latitude and longitude point in the
ocean and searches for the closest available data if the given latitude and longitude is not found in the dataset.

@author Darren Lie
@version August 28, 2023

"""

# Import Libraries
import xarray
import numpy as np
import pandas as pd

def maxDepth(df, lat, long, areaIncr):
    i = 0
    err = areaIncr*i
    latBetween = df['Latitude'].between(lat - err, lat + err)
    longBetween = df['Longitude'].between(long - err, long + err)
    filtered_df = df.loc[latBetween & longBetween, ['Depth']]

    # if there is no data at the given lat and long, check surroundings for depth data, increase by 25% error everytime
    while filtered_df.empty:
        i += 0.25
        err = areaIncr*i
        latBetween = df['Latitude'].between(lat - err, lat + err)
        longBetween = df['Longitude'].between(long - err, long + err)
        filtered_df = df.loc[latBetween & longBetween,['Depth']]

    return int(filtered_df.max()['Depth'])

if __name__ == "__main__":
    train_dataset = "practiceData3Years.nc"
    ds = xarray.open_dataset(train_dataset)
    df = ds.to_dataframe()

    maxDepth(df, -80, -165, 1)

 