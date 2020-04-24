from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class AndreivggNet:
    @staticmethod
    def build(width, height, depth, classes):
        # dimensiunile de intrare trebuie sa satisfaca conditia de a fi ch last | input shape to be ch last. It should
        # be mentioned that this is a static method!

        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # convolutional layer composed out of a conv stage, activation stage with relu, a batch normalization stage
        # and a max poooling stage with dropout included, to be observed the 3 by 3 kernel size for conv filter and
        # for max pooling kernel also. dropout of 0.25 used for redundancy

        model.add(Conv2D(32, (3, 3), padding="same",
              input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))

        # the same thing as above, only this time 2 conv layers and just after then performing max pooling in order to
        # reduce the spatial dimensions
        #model.add(Conv2D(64, (3, 3), padding="same"))
        #model.add(Activation("relu"))
        #model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # same as above, the same filter sizes and everything. it can be remarked that the maxpooling size is just
        # 2x2 now, not 3x3 in order to not reduce too fast the dimensions

        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # a fully connected layer and then a activation layer and the batch normalization

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        # the softmax activation function is used for classification
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # CNN architecture, all the libraries are open source. CNN architecture derived from the VGG model architecture
        return model
