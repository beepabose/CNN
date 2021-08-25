import glob
import numpy as np
from PIL import Image
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D,Flatten
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping



a=[]
y=np.array([])

names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

for k in range(4):
    temp = './'+names[k]+'/*'
    #print(temp)
    files=glob.glob(temp)
   # print(files)
    for i,j in enumerate(files):
        
        #print(j)
        if (i==50):
            break
        img = Image.open(j) 
        arr=np.array(img)  #converts image into an array of pixels
        a.append(arr)
        y=np.append(y,k)
X=np.array(a)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
X_train = X_train.reshape(X_train.shape[0], 208, 176, 1)
X_test = X_test.reshape(X_test.shape[0], 208, 176, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data to help with the training
X_train /= 255
X_test /= 255

n_classes = 4
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

# building a linear stack of layers with the sequential model

model = Sequential()
# convolutional layer
model.add(Conv2D(10, kernel_size=(4,4), strides=(1,1), padding='same', activation='relu', input_shape=(208, 176,1)))
#maxpool layer
model.add(MaxPool2D(pool_size=(2,2)))
# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(50, activation='relu'))
# output layer
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam') #optimizer = sgd
early_stopping_monitor = EarlyStopping(patience=3)

# Fit the model
#model.fit(predictors, target, validation_split=0.3, nb_epoch=30, callbacks=[early_stopping_monitor],verbose=False)
hist=model.fit(X_train, Y_train, batch_size=25, epochs=100, validation_split=0.1,callbacks=[early_stopping_monitor])
#,verbose=False


model.summary()

model.evaluate(X_test,Y_test)

plt.plot(hist.history['loss'], 'orange', label='Loss')
plt.plot(hist.history['val_loss'], 'green', label='Val_loss')
plt.xlabel('Epochs')
plt.ylabel('Scores')
plt.legend()
plt.show()




