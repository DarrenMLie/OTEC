import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Dataloading example
chunk = 100000
datafile = 'practiceMonth.csv'
tp = pd.read_csv(datafile, chunksize=chunk,
                     usecols=['Probe', 'Latitude', 'Longitude', 'Month', 'Depth', 'Temperature'],
                     dtype={'Probe': 'str', 'Latitude': 'float32', 'Longitude': 'float32'})
prac_data = pd.concat(tp, ignore_index=True)
prac_data.head()
prac_data.dropna(inplace=True)
prac_data = prac_data[prac_data.Probe != "XBT"]
#prac_data = prac_data[prac_data.Temperature > -3]

# histogram
num_bins = 100
minLat = -10.0
maxLat = 10.0
minLong = -10
maxLong = 10
minDep = 0
maxDep = 15
# latitude = 64.6667
# longitude = 8.6492
prac_data = prac_data.loc[(minLat < prac_data.Latitude) & (prac_data.Latitude < maxLat)]
prac_data = prac_data.loc[(minLong < prac_data.Longitude) & (prac_data.Longitude < maxLong)]
prac_data = prac_data.loc[(minDep < prac_data.Depth) & (prac_data.Depth < maxDep)]

# x is depth, y is temperature
print(prac_data)
y = prac_data["Temperature"]
print (y)
plt.hist(y, 100)
  
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
  
plt.title('Depth Temperature Histogram function Example\n\n',
          fontweight ="bold")
  
plt.show()

#Plotting example
# fig, ax = plt.subplots(2,3, dpi = 100, squeeze = False)
# ax[0,0].hist(prac_data["Latitude"])
# ax[0,0].set_title("Latitude")

# ax[0,1].hist(prac_data["Longitude"])
# ax[0,1].set_title("Longitude")

# ax[0,2].hist(prac_data["Month"])
# ax[0,2].set_title("Month")


# ax[1,0].hist(prac_data["Depth"])
# ax[1,0].set_title("Depth")


# ax[1,1].hist(prac_data["Temperature"])
# ax[1,1].set_title("Temperature")

#Example removing columns
# remove_cols = ['Probe','Day', 'Year']
# df = prac_data.drop(remove_cols, axis=1)
# #prac_data = prac_data.to_numpy()
# data = df.to_numpy()
# X = data[:, :-1]
# X = X[:,1:]
# y = data[:, -1][:, None]