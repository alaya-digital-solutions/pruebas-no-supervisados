# import os
# import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from random import randint

from tensorflow import keras
from livelossplot import PlotLossesKeras
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint)
from src.module import (get_data, plot_n, image_normalization)

# cargar imagenes
activos = []
pasivos = []

PATHACT = "data/images/activo/"
PATHPAS = "data/images/pasivo/"

activos += get_data(PATHACT, 9000, (128, 128))
pasivos += get_data(PATHPAS, 18000, (128, 128))

img_height, img_width = activos[0].shape[:2]

# msotrar algunos de los estados
plot_n(pasivos, n=2)
plot_n(activos, n=2)

# normalizar imagen
activos = image_normalization(activos, img_height, img_width)
pasivos = image_normalization(pasivos, img_height, img_width)

# pseudo etiquetas a comprobar con el autoencoder
y_activos = np.ones(len(activos))
y_pasivos = np.zeros(len(pasivos))

# división de los datos de activos
n = len(activos)
train_n = round(n*0.8)

activos_train = activos[:train_n, ...]
y_activos_train = y_activos[:train_n]
activos_test = activos[train_n:n, ...]
y_activos_test = y_activos[train_n:n]

# división de los datos de pasivos
n = len(pasivos)
print(n)
train_n = round(n*0.8)
pasivos_train = pasivos[:train_n, ...]
y_pasivos_train = y_pasivos[:train_n]
pasivos_test = pasivos[train_n:n, ...]
y_pasivos_test = y_pasivos[train_n:n]

# concatenar y ordenar
x_train = np.concatenate((activos_train, pasivos_train), axis=0)
y_train = np.concatenate((y_activos_train, y_pasivos_train))
x_train, y_train = shuffle(x_train, y_train)
x_test = np.concatenate((activos_test, pasivos_test), axis=0)
y_test = np.concatenate((y_activos_test, y_pasivos_test))
x_test, y_test = shuffle(x_test, y_test)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# empezar a entrenar una red
input_img = keras.Input(shape=(img_height, img_width, 1))
x = keras.layers.Conv2D(16, (3, 3), activation='relu',
                        padding='same')(input_img)
x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = keras.layers.Conv2D(2, (3, 3), activation='relu', padding='same')(x)
encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 16*32*64) i.e. 128-dimensional

x = keras.layers.Conv2D(2, (3, 3), activation='relu', padding='same')(encoded)
x = keras.layers.UpSampling2D((2, 2))(x)
x = keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.UpSampling2D((2, 2))(x)
x = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.UpSampling2D((2, 2))(x)
decoded = keras.layers.Conv2D(
    1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# hiperparametros
early_stopping_monitor = EarlyStopping(monitor="val_loss", patience=20)
bs = 64
epochs = 512
checkpoint = ModelCheckpoint(
    'ae.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=bs,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[PlotLossesKeras(), early_stopping_monitor,
                           checkpoint])

# encoded model
encoder = keras.Model(input_img, encoded)
encoder.save('encoder.h5')

# hacer carga de modelo encoder
encoder = load_model('encoder.h5')
autoencoder = load_model('ae.h5')

# evaluar en testing
decoded_imgs = autoencoder.predict(x_test)
decoded_imgs.shape


# ver reconstruciones visiblemente
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(128, 128))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(128, 128))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

for i in range(2):
    plt.imshow(decoded_imgs[randint(0, len(decoded_imgs))].reshape(
        128, 128), cmap='gray')
    plt.show()

# sacar representaciones del conjunto de test
encoded_imgs = encoder.predict(x_test)
n = 10
plt.figure(figsize=(20, 8))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(encoded_imgs[i].reshape((16, 16*4)).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
