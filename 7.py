import keras 
from keras.datasets import mnist 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten 
from keras.layers import Conv2D, MaxPooling2D 
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train.shape
x_train[0].shape
plt.imshow(x_test[0])

'''
rs = np.resize(x_test[0],(40,40))
plt.imshow(rs)

rs.shape
'''

x_train.shape
y_train.shape

x_test.shape
y_test.shape

img_rows, img_cols = 28, 28 

if K.image_data_format() == 'channels_first': 
   x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2]) 
   x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols) 
   input_shape = (1, img_rows, img_cols) 
else: 
   x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1) 
   x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1) 
   input_shape = (img_rows, img_cols, 1) 
   
   

x_train.shape
y_train.shape

x_test.shape
y_test.shape


plt.imshow(x_test[0])


x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 
x_train /= 255 
x_test /= 255 

plt.imshow(x_test[0])


y_train = keras.utils.to_categorical(y_train, 10) 
y_test = keras.utils.to_categorical(y_test, 10)
y_test[0]

model = Sequential() 
model.add(Conv2D(32, kernel_size = (3, 3),  
   activation = 'relu', input_shape = input_shape)) 
model.add(MaxPooling2D(pool_size = (2, 2))) 
model.add(Conv2D(64, (3, 3), activation = 'relu')) 
model.add(MaxPooling2D(pool_size = (2, 2))) 
model.add(Dropout(0.1)) 
model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))

model.compile(loss = keras.losses.categorical_crossentropy, optimizer= 'adam', metrics=['accuracy'])

model.summary()
model.fit(
   x_train, y_train, 
   batch_size = 128, 
   epochs = 12, 
   verbose = 1, 
   validation_data = (x_test, y_test)
)

model
score = model.evaluate(x_test, y_test, verbose = 0) 

print('Test loss:', score[0]) 
print('Test accuracy:', score[1])

pred = model.predict(x_test) 
pred
pred = np.argmax(pred, axis = 1)
pred

label = np.argmax(y_test,axis = 1) 

print(pred) 
print(label)



from sklearn.metrics import classification_report

print(classification_report(label, pred))
