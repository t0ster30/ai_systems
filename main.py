import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

dataframe = pd.read_csv("iris.csv", eader=None)
dataset = dataframe.values

X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

dummy_y = to_categorical(encoded_Y)

model = Sequential()
model.add(Dense(8, input_dim=4, activation="relu"))
model.add(Dense(3, activation="softmax"))

model.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    merics=["accuracy"]
)

history = model.fit(
    X,
    dummy_y,
    epochs = 150,
    batch_size = 10,
    validation_split = 0.2,
    verbose = 1
)

