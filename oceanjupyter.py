from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import numpy as np

# wandb.init(project='OTEC', config=config)
dataset = pd.read_csv('practiceData.csv')
X = dataset.iloc[:, 2:-1].values
Y = dataset.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_train = sc.fit_transform(X_train)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer='adam', loss='categorical_crossentropy',
            metrics=['accuracy'])
ann.fit(X_train, Y_train, batch_size=32, epochs=20)
