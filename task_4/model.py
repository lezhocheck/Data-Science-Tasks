# U-Net architecture for image segmentation
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from keras import Model
from keras.engine.base_layer import Layer
import numpy as np
import os
import cv2
from pickle import dump


EPOCHS = 20

IMG_SIZE = 256
IMG_CHANNELS = 3

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
TRAIN_PATH = os.path.join(DIR_PATH, 'data/feed/train/')
TEST_PATH = os.path.join(DIR_PATH, 'data/feed/test/')


# loads images from specified directory and prepares them for training
class DataConverter:
    def __init__(self) -> None:
        self._train_folders_names = next(i[1] for i in os.walk(TRAIN_PATH))
        self._test_folders_names = next(i[1] for i in os.walk(TEST_PATH))
        
    def get_train_data(self) -> tuple[np.array, np.array]:
        return self._get_data(TRAIN_PATH, self._train_folders_names)    
    
    def _get_data(self, dir: str, subdir_names: list[str]) -> tuple[np.array, np.array]:
        x = np.zeros((len(subdir_names), IMG_SIZE, IMG_SIZE, IMG_CHANNELS), dtype=np.float64)
        y = np.zeros((len(subdir_names), IMG_SIZE, IMG_SIZE, 1), dtype=np.bool_)
        for index, folder in enumerate(subdir_names):
            folder_path = os.path.join(dir, folder)
            img = cv2.imread(folder_path + '/img/' + folder + '.jpg')[:,:,:IMG_CHANNELS]  
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            x[index] = img
            mask = cv2.imread(folder_path + '/mask/' + folder + '.jpg')
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            mask = mask / 255.0
            mask[mask >= 0.5] = 1
            mask[mask < 0.5] = 0
            y[index] = np.expand_dims(mask, axis=-1) 
        return x, y

    def get_test_data(self) -> np.array:
        x, _ = self._get_data(TEST_PATH, self._test_folders_names)
        return x
    
    def get_test_labeled_data(self) -> tuple[np.array, np.array]:
        return self._get_data(TEST_PATH, self._test_folders_names)


def build_model(input_layer: Input) -> Layer:
    conv1 = Conv2D(16, 3, activation='relu', kernel_initializer='he_normal', padding='same')(input_layer)
    # add dropout regularization to prevent overfitting
    conv1 = Dropout(0.1)(conv1)
    conv1 = Conv2D(16, 3, activation='relu', kernel_initializer='he_normal', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=2)(conv1)

    conv2 = Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = Dropout(0.1)(conv2)
    conv2 = Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=2)(conv2)
    
    conv3 = Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=2)(conv3)
    
    conv4 = Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=2)(conv4)
    
    conv5 = Conv2D(256, 3, activation='relu', kernel_initializer='he_normal', padding='same')(pool4)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2D(256, 3, activation='relu', kernel_initializer='he_normal', padding='same')(conv5)

    merge6 = Conv2DTranspose(128, 2, strides=2, padding='same')(conv5)
    merge6 = concatenate([merge6, conv4])
    conv6 = Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same')(merge6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(128, 3, activation='relu', kernel_initializer='he_normal', padding='same')(conv6)
    
    merge7 = Conv2DTranspose(64, 2, strides=2, padding='same')(conv6)
    merge7 = concatenate([merge7, conv3])
    conv7 = Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(merge7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(64, 3, activation='relu', kernel_initializer='he_normal', padding='same')(conv7)
    
    merge8 = Conv2DTranspose(32, 2, strides=2, padding='same')(conv7)
    merge8 = concatenate([merge8, conv2])
    conv8 = Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(merge8)
    conv8 = Dropout(0.1)(conv8)
    conv8 = Conv2D(32, 3, activation='relu', kernel_initializer='he_normal', padding='same')(conv8)
    
    merge9 = Conv2DTranspose(16, 2, strides=2, padding='same')(conv8)
    merge9 = concatenate([merge9, conv1], axis=3)
    conv9 = Conv2D(16, 3, activation='relu', kernel_initializer='he_normal', padding='same')(merge9)
    conv9 = Dropout(0.1)(conv9)
    conv9 = Conv2D(16, 3, activation='relu', kernel_initializer='he_normal', padding='same')(merge9)
    
    return Conv2D(1, 1, activation='sigmoid')(conv9)


def start() -> None:
    input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS))
    outputs = build_model(input_layer)
    model = Model(inputs=input_layer, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    conv = DataConverter()
    x_train, y_train = conv.get_train_data()
    results = model.fit(x_train, y_train, validation_split=0.2, batch_size=16, epochs=EPOCHS)
    model.save(os.path.join(DIR_PATH, 'soil_erosion_model.h5'))

    with open(os.path.join(DIR_PATH, 'history.save'), 'wb') as hist:
        dump(results.history, hist)


if __name__ == '__main__':
    start()