import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dataloading example
chunk = 100000
datafile = 'practiceMonth.csv'
tp = pd.read_csv(datafile, chunksize=chunk,
                 usecols=['Probe', 'Latitude', 'Longitude',
                          'Month', 'Depth', 'Temperature'],
                 dtype={'Probe': 'str', 'Latitude': 'float32', 'Longitude': 'float32'})
prac_data = pd.concat(tp, ignore_index=True)
prac_data.head()
prac_data.dropna(inplace=True)

month = 7
num_bins = 100
minDep = 0
maxDep = 10
depthchunk = 10
minLong = -1
maxLong = 1
minLat = -1
maxLat = 1
prac_data = prac_data[prac_data.Probe == "CTD"]
#prac_data = prac_data[prac_data.Temperature < 35]
#prac_data = prac_data[prac_data.Temperature > -3]
prac_data = prac_data[prac_data.Month == month]
prac_data = prac_data.loc[(minDep < prac_data.Depth)
                          & (prac_data.Depth < maxDep)]
prac_data = prac_data.loc[(minLat < prac_data.Latitude)
                          & (prac_data.Latitude < maxLat)]
prac_data = prac_data.loc[(minLong < prac_data.Longitude)
                          & (prac_data.Longitude < maxLong)]
sigmas = []
for i in range(0, maxDep//depthchunk):
    depthRange = [1+(depthchunk*i), depthchunk+(depthchunk*i)]
    temp_data = prac_data.loc[(depthRange[0] < prac_data.Depth) & (
        prac_data.Depth < depthRange[1])]
    print(temp_data)
    temp_data.to_csv(
        f'Data_month_{month}_long_{(minLong+maxLong)/2}_dep_{depthRange[0]}_{depthRange[1]}.csv')

    # x is depth, y is temperature
    # print(prac_data)
    y = temp_data["Temperature"]
    sigmas.append(np.std(temp_data["Temperature"]))
    #print (y)
    plt.hist(y, 100)
    plt.xlabel('Temperature (C)')
    plt.ylabel('Frequency')
    plt.title(f'Depth: {depthRange[0]} to {depthRange[1]} Longitude: {minLong} to {maxLong} Histogram\n',
              fontweight="bold")
    plt.savefig(
        f'Histogram_Month_{month}_long_{(minLong+maxLong)/2}_dep_{depthRange[0]}_{depthRange[1]}.png')
    plt.clf()

print(sigmas)
plt.plot(sigmas)
plt.xlabel('Depth Range')
plt.ylabel('Standard Deviation')
plt.title(f'Depth: {depthRange[0]} to {depthRange[1]} Longitude: {minLong} to {maxLong} Histogram\n',
          fontweight="bold")
plt.savefig(
    f'STD_Month_{month}_long_{(minLong+maxLong)/2}_dep_{minDep}_{maxDep}.png')


# Plotting example
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

# Example removing columns
# remove_cols = ['Probe','Day', 'Year']
# df = prac_data.drop(remove_cols, axis=1)
# #prac_data = prac_data.to_numpy()
# data = df.to_numpy()
# X = data[:, :-1]
# X = X[:,1:]
# y = data[:, -1][:, None]
