# LOCK VERSIONS OF KERAS AND TENSORFLOW TO 2.12.1

import pandas as pd
import os
import kagglehub

path = kagglehub.dataset_download("uciml/iris")
print("Path to dataset files:", path)
csv_path = os.path.join(path, "Iris.csv")

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

dataframe = pd.read_csv(csv_path)
print(dataframe.head())
dataset = dataframe.values

X = dataset[:, 1:5].astype(float)
Y = dataset[:, 5]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

dummy_y = to_categorical(encoded_Y)

model = Sequential()
model.add(Dense(8, input_dim=4, activation="relu"))
model.add(Dense(3, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X,
    dummy_y,
    epochs=300,
    batch_size=30,
    validation_split=0.2,
    verbose=1
)

