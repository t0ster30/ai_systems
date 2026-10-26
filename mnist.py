import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras import metrics
from keras.utils import to_categorical
from keras.models import Sequential

num_threads = os.cpu_count()

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(f'Форма обучающих изображений: {train_images.shape}')
print(f'Форма обучающих меток: {train_labels.shape}')
print(f'Форма тестовых изобраджений: {test_images.shape}')
print(f'Форма тестовых меток: {test_labels.shape}')
print(f'Пример меток: {train_labels[:10]}')

plt.figure(figsize=(4,4))
plt.imshow(train_images[0],cmap=plt.cm.binary)
plt.title(f'Пример изображения: {train_labels[0]}')
plt.axis('off')
plt.show()

# Нормализация пикселей
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
print(f'Диапазон значений пикселей после нормализации: [{np.min(train_images)}, {np.max(train_images)}]')

# Преобразование меток в векторы
train_labels_one_hot = to_categorical(train_labels, 10)
test_labels_one_hot = to_categorical(test_labels, 10)
print(f'Форма one-hot меток (обучение): {train_labels_one_hot.shape}')
print(f'Форма one-hot меток (тест): {test_labels_one_hot.shape}')
print(f'Пример one-hot метки для "5": {train_labels_one_hot.shape}')

# Создание модели
model_base = Sequential()
model_base.add(layers.Flatten(input_shape=(28,28)))
model_base.add(layers.Dense(256, activation='relu'))
model_base.add(layers.Dense(10, activation='softmax'))
model_base.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
model_base.summary()

BATCH_SIZE = 128
train_ds = (
    tf.data.Dataset.from_tensor_slices((train_images, train_labels_one_hot))
    .shuffle(10000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)   # готовить следующие батчи, пока идёт обучение
)
test_ds = (
    tf.data.Dataset.from_tensor_slices((test_images, test_labels_one_hot))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
history_base = model_base.fit(train_ds,
                              epochs=5,
                              validation_data=test_ds,
                              verbose=1)
test_loss, test_accuracy = model_base.evaluate(test_images, test_labels_one_hot, verbose=0)
print(f'\n--- Оценка базовой модели на тестовых данных ---')
print(f'Точность (Accuracy): {test_accuracy*100:.2f}%')


def plot_training_history(history, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # График потерь
    ax1.plot(history.history['accuracy'], label='Обучение (Accuracy)')
    ax1.plot(history.history['val_accuracy'], label='Валидация (Accuracy)')
    ax1.set_title(f'{title} - Точность')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Точность')
    ax1.legend()

    ax2.plot(history.history['loss'], label='Обучение (Loss)')
    ax2.plot(history.history['val_loss'], label='Валидация (Loss)')
    ax2.set_title(f'{title} - Потери')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Потери')
    ax2.legend()

    plt.tight_layout()
    plt.show()


plot_training_history(history_base, 'Базовая модель (5 эпох)')


def create_improved_model():
    model = Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
    return model


model_improved = create_improved_model()
history_improved = model_improved.fit(train_images, train_labels_one_hot,
                              epochs=20,
                              batch_size=128,
                              validation_data=(test_images, test_labels_one_hot),
                              verbose=1)
test_loss_imp, test_accuracy_imp = model_improved.evaluate(
    test_images, test_labels_one_hot, verbose=0
)
print(f'\n--- Оценка улучшенной модели на тестовых данных ---')
print(f'Точность (Accuracy): {test_accuracy_imp*100:.2f}%')

plot_training_history(history_improved, 'Модель RMSprop, 20 эпох')

if test_accuracy_imp >= 0.95:
    print('\nЦель достигнута: точность >= 95%')
else:
    print(f'\n Цель не достигнута. Текущая точность: {test_accuracy_imp*100:.2f}%. Продолжайте экспериментировать!')

model_rmsprop = create_improved_model()
model_rmsprop.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

history_rmsprop = model_rmsprop.fit(
    train_ds,
    epochs=20,
    validation_data=test_ds,
    verbose=1
)

test_loss_rms, test_accuracy_rms = model_rmsprop.evaluate(test_images, test_labels_one_hot, verbose=0)
print(f'\n--- Оценка модели RMSprop на тестовых данных ---')
print(f'Точность (Accuracy): {test_accuracy_rms*100:.2f}%')

plot_training_history(history_rmsprop, 'Улучшенная модель (20 эпох)')

print(f'\n--- Сравнение точностей ---')
print(f'  Улучшенная модель (Adam)  {test_accuracy_imp*100:.2f}%')
print(f'  Модель (RMSprop):         {test_accuracy_rms*100:.2f}%')