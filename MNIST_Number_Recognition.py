# First things first, lets import packages I will be using

import tensorflow as tf
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

# load data from MNIST and store train data and test data differently
# We use the train data to configure the model and use the test data for model assessment
(x_train, y_train), (x_test, y_test)  = tf.keras.datasets.mnist.load_data()

# Let's check the shape of the MNIST dataset

x_train.shape

# Now that we know what kind of format is our data and its size
# Let's do some data pre-processing to be ready to feed in the CNN we create

# The input in Keras has to be 4 dimensional so first we reshape the images
# Currently, the images are grayscale 2d arrays of the pixels

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# The images need to be in uniform aspect ratio - this simply means that the image needs to be a square
# The images are already a square 28 x 28 hence we do not need to do anything

# We then do grayscale normalization to ensure that each pixel has a similar data distribution
# It also makes convergence faster
# We do this by simply dividing by 255, however, we need to ensure the values are float to get decimals after division

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# We divide by 255 because it is the max value a pixel can have
x_train /= 255
x_test /= 255

# Let's check the shape now
print('x_train shape:', x_train.shape)

# Sometimes the dimensionality of the data is also reduced. However, in our case we only have one dimension
# because it is grayscale so we don't have to do anything

# We have 60,000 training images, but guess what we can make the data even larger by doing some data-augmentation!!
# This has some advantages, first of all, the more data we have the better! These models are hungry and the more
# You feed them, generally they perform better
# Secondly, we avoid over-fitting because we are making small transformations in the images which increases the
# variance. Thus, we end up creating a robust model by doing data augmentation
# I use ImageDataGenerator and only transform images in certain ways

data_generator = ImageDataGenerator(
                    featurewise_center=False,
                    samplewise_center=False,
                    featurewise_std_normalization=False,
                    samplewise_std_normalization=False,
                    zca_whitening=False,
                    rotation_range=12, # Randomly rotate images from 0-180 degrees
                    zoom_range=0.15,  # Randomly zoom images
                    width_shift_range=0.15,
                    height_shift_range=0.15,
                    horizontal_flip=False,  # randomly flip images, but it does not make sense for us to do this
                    vertical_flip=False)

data_generator.fit(x_train)  # fit it in the training data

# Let's build the model
# I will be using the Keras Sequential API where we just add one layer at a time
model = Sequential()

# The first layer we add is the convolutional  layer
# I chose to add two layers of 32 filters and a kernel size of 5,5

model.add(Conv2D(filters=32, kernel_size=(5,5),padding='Same', activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(filters=32, kernel_size=(5,5),padding='Same', activation='relu', input_shape=(28,28,1)))

# Next we add the pooling layer which sort of down samples the filter
# pooling layer reduces over-fitting and reduces computational costs
model.add(MaxPool2D(pool_size=(2,2)))

# Next we add a regularization method called dropout. Dropout basically ignores some of the neurons
# This forces the network to learn in a distributed way
model.add(Dropout(0.25))

# I use the 'ReLU' activation function. ReLU basically adds some non-linearity into the model
# ReLU reduces computational costs and avoids gradient killing. However, it has a disadvantage of not being
# 'Zero centered'

# I add more layers in our model so that it can "learn" the features of the different digits

model.add(Conv2D(filters=64, kernel_size=(2,2),padding='Same',
                 activation ='relu'))
model.add(Conv2D(filters=64, kernel_size = (2,2),padding='Same',
                 activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

# Now we flatten all the learned features into a one dimensional vector.
# This layer can be thought of as the "summary" layer as it combines all learned features from the previous layers

model.add(Flatten())

# Lastly, we add a dense layer specifically to output the probability the digit falls in the specific category

model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

# Time to Optimize!!
# First we choose a loss function which we will use to iteratively improve the CNN
# In non-binary classification tasks we use the categorical cross-entropy loss function
# For optimizer I chose RMSprop with default values because it is usually faster and effective
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Compile the model
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Next we choose the learning rate: the rate at which our CNN will learn
# Normally learning rates that decrease with time are used because they lead to convergence faster
# Also, they are less computationally expensive

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,   # Reduce LR by "factor" if no improvement in 3 epochs
                                            verbose=1,
                                            factor=0.5,   # Reduce LR by half
                                            min_lr=0.00001)

epochs = 1 # Turn epochs to 30 to get 0.9967 accuracy
batch_size = 86

# Fitting the model
history = model.fit_generator(data_generator.flow(x_train,y_train, batch_size=batch_size),
                              epochs=epochs, validation_data=(x_test,y_test),
                              verbose=2, steps_per_epoch=x_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])
                             
