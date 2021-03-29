import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, add, BatchNormalization, Activation, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import imgaug.augmenters as iaa
import random


epochs = 50
batch_size = 192
kernel_size = (3,3)
seed = 32

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

num_labels = len(np.unique(Y_train))

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)


def data_augmentation(X_train, Y_train, n):
     data_sample = random.sample(list(zip(X_train, Y_train)), n)

     i = 0

     X_train_augmented = X_train.copy()
     Y_train_augmented = Y_train.copy()

     rotate = iaa.Affine(rotate=-50)
     gaussian_noise = iaa.AdditiveGaussianNoise(10, 20)
     crop = iaa.Crop(percent=(0, 0.3))
     shear = iaa.Affine(shear=(0, 40))
     flip_hr = iaa.Fliplr(p=1.0)
     flip_vr = iaa.Flipud(p=1.0)
     contrast = iaa.GammaContrast(gamma=2.0)
     scale_im = iaa.Affine(scale={"x": (1.5, 1.0), "y": (1.5, 1.0)})

     for image, label in data_sample:
         i = i + 1
         if i % 100 == 0:
             print(i)
         label = label.reshape((1, 10))

         rotated_image = rotate.augment_image(image)
         noise_image = gaussian_noise.augment_image(image)
         crop_image = crop.augment_image(image)
         shear_image = shear.augment_image(image)
         flip_hr_image = flip_hr.augment_image(image)
         flip_vr_image = flip_vr.augment_image(image)
         contrast_image = contrast.augment_image(image)
         scale_image = scale_im.augment_image(image)

         aug_images = np.concatenate([rotated_image, noise_image, crop_image, shear_image,
                                      contrast_image, scale_image, flip_hr_image, flip_vr_image], axis=0)

         aug_images = aug_images.reshape((-1, 32, 32, 3))
         aug_shape = aug_images.shape[0]
         aug_labels = np.tile(label, (aug_shape, 1))

         X_train_augmented = np.concatenate([X_train_augmented, aug_images], axis=0)
         Y_train_augmented = np.concatenate([Y_train_augmented, aug_labels], axis=0)


     return X_train_augmented, Y_train_augmented


X_train, Y_train = data_augmentation(X_train, Y_train, 6250)


X_train= X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

inputs = Input(shape=(32, 32, 3))

def InceptionResNetModule(input, n_filters, kernel_size):

    path_a = Conv2D(filters=n_filters/32, kernel_size=(1,1),
               kernel_initializer=tf.keras.initializers.he_uniform(seed=seed))(input)
    path_a = BatchNormalization(axis=-1)(path_a)
    path_a = Activation('elu')(path_a)

    path_b = Conv2D(filters=n_filters, kernel_size=kernel_size, padding='same',
               kernel_initializer=tf.keras.initializers.he_uniform(seed=seed))(input)
    path_b = BatchNormalization(axis=-1)(path_b)
    path_b = Activation('elu')(path_b)

    path_b = Conv2D(filters=n_filters, kernel_size=kernel_size, padding='same',
               kernel_initializer=tf.keras.initializers.he_uniform(seed=seed))(path_b)
    path_b = BatchNormalization(axis=-1)(path_b)

    shortcut = Conv2D(filters=n_filters, kernel_size=(1,1),  activation='elu',
               kernel_initializer=tf.keras.initializers.he_uniform(seed=seed))(input)
    shortcut = BatchNormalization(axis=-1)(shortcut)
    shortcut = Activation('elu')(shortcut)

    path_b = Activation('elu')(add([shortcut, path_b]))

    path_c = Conv2D(filters=n_filters/32, kernel_size=(1, 1),
                    kernel_initializer=tf.keras.initializers.he_uniform(seed=seed))(input)
    path_c = BatchNormalization(axis=-1)(path_c)
    path_c = Activation('elu')(path_c)

    output = concatenate([path_a, path_b, path_c])

    output = MaxPooling2D(pool_size=(2, 2))(output)
    y = Dropout(0.2)(output)

    return y




y = Conv2D(filters=32, kernel_size=(1,1),
               kernel_initializer=tf.keras.initializers.he_uniform(seed=seed))(inputs)
y = InceptionResNetModule(y, 32, kernel_size)
y = InceptionResNetModule(y, 64, kernel_size)
y = InceptionResNetModule(y, 128, kernel_size)
y = Flatten()(y)
y = Dropout(0.2)(y)
y = Dense(128, activation='elu')(y)
y = Dropout(0.2)(y)
outputs = Dense(num_labels, activation='softmax')(y)


model = Model(inputs=inputs, outputs=outputs)
model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)
loss, acc = model.evaluate(X_test, Y_test, batch_size=batch_size)



































