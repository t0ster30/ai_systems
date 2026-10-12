import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

import certifi
from pathlib import Path


def plot_training_history(history, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # График потерь
    ax1.plot(history.history['loss'], label='Обучение (loss')
    ax1.plot(history.history['val_loss'], label='Валидация (val loss)')
    ax1.set_title(f'{title} - Потери')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Точность')

    ax2.plot(history.history['accuracy'], label='Обучение (Accuracy)')
    ax2.plot(history.history['val_accuracy'], label='Валидация (Val Accuracy)')
    ax2.set_title(f'{title} - Точность')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Точность')
    ax2.legend()

    plt.tight_layout()
    plt.show()


# Read corporate certificate content
# corp_cert = Path(r"C:\Users\29415403\Documents\certs.pem").read_bytes()
#
# # Merge into certifi bundle (append mode, avoid overwriting existing public CA certificates)
# with open(certifi.where(), "ab") as f:
#     f.write(b"\n" + corp_cert)
#
# print("Merged to:", certifi.where())

dataframe = pd.read_csv("sonar.csv", header=None)

print(dataframe.shape)
print(dataframe.head())
dataset = dataframe.values
print(dataset)
X = dataset[:, 0:60].astype(float)
Y_labels = dataset[:, 60] # строковые метки R или M

print(f"Форма X: {X.shape}")
print(f"Форма Y_labels: {Y_labels.shape}")
print(f"Уникальные метки: {np.unique(Y_labels)}")

# sonar = fetch_ucirepo(id=151)
#
# print(sonar.metadata)
# print(sonar.variables)
# print(sonar.data.features)
# print(sonar.data.targets)
# X = sonar.data.features
# Y_labels = sonar.variables # строковые метки


# Кодирование строковых меток в числовые 0 и 1
encoder = LabelEncoder()
encoder.fit(Y_labels)
Y_encoded = encoder.transform(Y_labels) # Теперь Y_encoded содержит 0 и 1

print(f"Закодированные метки (первые 10): {Y_encoded[:10]}")

# Создание модели
model_base = Sequential()
# Первый скрытый слой: 60 нейронов, ReLU, 60 признаков на вход
model_base.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
# Компиляция модели
model_base.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Просмотр архитектуры
model_base.summary()
# Обучение модели (verbose=1 для прогресса)
history_base = model_base.fit(X, Y_encoded, epochs=100, batch_size=10, validation_split=0.1, verbose=1)

model_exp1 = Sequential()
model_exp1.add(Dense(30, input_dim=60, kernel_initializer='normal', activation='relu'))
model_exp1.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
model_exp1.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[metrics.Accuracy(), metrics.Precision()]
)
model_exp1.summary()
history_exp1 = model_exp1.fit(X, Y_encoded, epochs=100, batch_size=10, validation_split=0.1, verbose=1)
plot_training_history(history_exp1, "Эксперимент 1 (60->30->1)")

# Создание модели
model_exp2 = Sequential()
model_exp2.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
model_exp2.add(Dense(15, kernel_initializer='normal', activation='relu'))
model_exp2.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
model_exp2.compile(
  optimizer='adam',
  loss='binary_crossentropy',
  metrics=['accuracy']
)
model_exp2.summary()
history_exp2 = model_exp2.fit(X, Y_encoded, epochs=100, batch_size=10, validation_split=0.1, verbose=1)
plot_training_history(history_exp2, "Эксперимент 2 (60->60->15->1)")