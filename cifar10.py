import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.datasets import cifar10
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras import metrics
from keras.utils import to_categorical

# Загрузка данных
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Просмотр размерности
print(f'Форма обучающих изображений: {X_train.shape}')
print(f'Форма обучающих меток: {y_train.shape}')
print(f'Форма тестовых изоображений: {X_test.shape}')
print(f'Форма тестовых меток: {y_test.shape}')

# Уникальные метки
num_classes = len(np.unique(y_train))
print(f'Количество классов: {num_classes}')
print(f'Уникальные метки: {np.unique(y_train).flatten()}')

# Названия классов визуализации
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Просмотр примера изображения
plt.figure(figsize=(6, 6))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    # В CIFAR-10 изображения уже идут 32x32, 3 с uint8 [0,255]
    plt.imshow(X_train[i])
    plt.title(f'{class_names[y_train[i][0]]}') # y_train[i] -- массив [метка]
    plt.axis('off')
plt.suptitle('Примеры изображений CIFAR-10')
plt.tight_layout()
plt.show()

# Првоерка диапазона значений пикселей
print(f'Диапазон значений пикселей в исходных данных: [{np.min(X_train)}, {np.max(X_train)}]')

# Гиперпараметры
batch_size = 32
num_epochs = 20
kernel_size = 3
pool_size = 2
conv_depth_1 = 32
conv_depth_2 = 64
drop_prob_1 = 0.25 # Dropout после пулинга
drop_prob_2 = 0.5  # Dropout в полносвязном слое
hidden_size = 512

# Предобработка
# 1. Преобразуем типы данных и нормализуем в [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# 2. Преобразуем метки в векторы
Y_train = to_categorical(y_train, num_classes)
Y_test = to_categorical(y_test, num_classes)

print(f'Форма обработанных обучающих изображений: {X_train.shape}')
print(f'Форма обработанных обучающих меток (one-hot): {Y_train.shape}')
print(f'Форма обработанных тестовых изоображений: {X_test.shape}')
print(f'Форма обработанных тестовых меток (one-hot): {Y_test.shape}')

# Создание и компиляция базовой модели с Dropout
# Входной тензор; в Keras aормат данных обычно channels_last (batch, height, width, channels)
input_img = Input(shape=(32, 32, 3))

# Блок 1: Conv -> Conv -> Pool --> Dropout
conv_1 = Conv2D(conv_depth_1, (kernel_size, kernel_size), activatino='relu', padding='same')(input_img)
conv_2 = Conv2D(conv_depth_1, (kernel_size, kernel_size), activatino='relu', padding='same')(conv_1)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
drop_1 = Dropout(drop_prob_1)(pool_1)

# Блок 2: Conv -> Conv -> Pool --> Dropout
conv_3 = Conv2D(conv_depth_1, (kernel_size, kernel_size), activatino='relu', padding='same')(drop_1)
conv_4 = Conv2D(conv_depth_1, (kernel_size, kernel_size), activatino='relu', padding='same')(conv_3)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
drop_3 = Dropout(drop_prob_1)(pool_2)