
#!/usr/bin/env python
# coding: utf-8

# Libraries
import xarray
import numpy as np
import pandas as pd


def maxDepth(df, lat, long, areaIncr):
    i = 0
    err = areaIncr*i
    filtered_df = df.loc[(df['Latitude'].between(lat - err, lat + err)) 
           & (df['Longitude'].between(long - err, long + err)),['Depth']]
    while filtered_df.empty:
        i += 0.25
        err = areaIncr*i
        filtered_df = df.loc[(df['Latitude'].between(lat - err, lat + err)) 
            & (df['Longitude'].between(long - err, long + err)),['Depth']]

    # print(filtered_df)
    # print("test")
    # print(filtered_df.max()["Depth"])
    return int(filtered_df.max()["Depth"])

if __name__ == "__main__":
    train_dataset = "practiceData3Years.nc"
    ds = xarray.open_dataset(train_dataset)
    df = ds.to_dataframe()

    maxDepth(df, 0, 0, 1)

 