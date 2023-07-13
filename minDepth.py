
#!/usr/bin/env python
# coding: utf-8

# Libraries
import xarray
import numpy as np
import pandas as pd

def minDepth(df, lat, long, areaIncr):
    i = 0.5
    err = areaIncr*i
    filtered_df = df.loc[(df['Latitude'].between(lat - err, lat + err)) 
           & (df['Longitude'].between(long - err, long + err)),['Depth']]
    
    # just make it 0.5 err each time since we approximate by a square area anyways
    # while filtered_df.empty:
    #     i += 0.25
    #     err = areaIncr*i
    #     filtered_df = df.loc[(df['Latitude'].between(lat - err, lat + err)) 
    #         & (df['Longitude'].between(long - err, long + err)),['Depth']]

    return int(filtered_df.min()["Depth"])

if __name__ == "__main__":
    train_dataset = "practiceData3Years.nc"
    ds = xarray.open_dataset(train_dataset)
    df = ds.to_dataframe()

    minDepth(df, 0, 0, 1)

 