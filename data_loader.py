import numpy
import pandas as pd
from pickle import dump
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def data_loader(datafile):
    """Function to load CSV data, remove unnecessary columns, split into training and validation datasets,
    randomize and save to disk
    """
    chunk = 100000
    test_size = 0.33

    tp = pd.read_csv(datafile, chunksize=chunk,
                     usecols=['Probe', 'Latitude', 'Longitude',
                              'Month', 'Depth', 'Temperature'],
                     dtype={'Probe': 'str', 'Latitude': 'float32', 'Longitude': 'float32'})
    prac_data = pd.concat(tp, ignore_index=True)
    prac_data.head()
    prac_data = prac_data[prac_data.Probe != "XBT"]
    prac_data = prac_data[prac_data.Temperature > -3]
    remove_cols = ['Probe']
    df = prac_data.drop(remove_cols, axis=1)
    df.dropna(inplace=True)
    del tp
    del prac_data
    gc.collect()
    df.sample(frac=1)
    data = df.to_numpy()
    del df
    gc.collect()
    X = data[:, :-1]
    y = data[:, -1][:, None]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size)
    # Build the scalers
    feature_scaler = StandardScaler().fit(X_train)
    target_scaler = StandardScaler().fit(y_train)
    dump(feature_scaler, open('featureScaler.pkl', 'wb'))
    dump(target_scaler, open('targetScaler.pkl', 'wb'))
    # Get scaled versions of the data
    X_train_scaled = feature_scaler.transform(X_train)
    y_train_scaled = target_scaler.transform(y_train)
    X_val_scaled = feature_scaler.transform(X_val)
    y_val_scaled = target_scaler.transform(y_val)
    # del X_train
    # del y_train
    # del X_val
    # del y_val
    del X
    del y
    gc.collect()
    numpy.savetxt('PracticeData20YearsRandTrain.csv', numpy.concatenate((X_train_scaled, y_train_scaled), axis=1),
                  delimiter=',')
    del X_train_scaled
    del y_train_scaled
    gc.collect()
    numpy.savetxt('PracticeData20YearsRandXVal.csv',
                  X_val_scaled, delimiter=',')
    numpy.savetxt('PracticeData20YearsRandYVal.csv',
                  y_val_scaled, delimiter=',')
    del y_val_scaled
    del X_val_scaled
    gc.collect()
    return X_val, y_val
