import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

#These parameters can be changed:
batch_size = 128
num_classes = 10
epochs = 1

#The Data Loading...
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#The Preprocessing of Inputs...
#The main difference of the ConvNet from DNN needs the multichannel input in comparison to the
# simple vector. As a result, the input data format should include the data about the channel (depth).
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

#The Preprocessing of Outputs...
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#The Model Construction...
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

#The Model Configuration...
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#The Model Training...
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=1,
          verbose=1,
          validation_data=(x_test, y_test))

#The Model Evaluation...
result = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', result[0])
print('Test accuracy:', result[1])