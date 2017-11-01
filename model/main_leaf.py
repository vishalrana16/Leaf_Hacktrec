import os,cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam


PATH = os.getcwd()
data_path = PATH + '/data'

data_dir_list = os.listdir(data_path)

img_rows=128
img_cols=128
num_channel=1
num_epoch=20

num_classes = 40

img_data_list=[]

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	for img in img_list:
		input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
		input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
		input_img_resize=cv2.resize(input_img,(128,128))
		img_data_list.append(input_img_resize)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print (img_data.shape)

if num_channel==1:
	if K.image_dim_ordering()=='th':
		img_data= np.expand_dims(img_data, axis=1) 
		print (img_data.shape)
	else:
		img_data= np.expand_dims(img_data, axis=4) 
		print (img_data.shape)
		
else:
	if K.image_dim_ordering()=='th':
		img_data=np.rollaxis(img_data,3,1)
		print (img_data.shape)
		


num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:12]=1
labels[12:25]=9
labels[25:41]=10
labels[41:53]=11
labels[53:66]=12
labels[66:78]=13
labels[78:88]=14
labels[88:98]=15
labels[98:103]=16
labels[103:115]=17
labels[115:128]=18
labels[128:138]=1
labels[138:150]=19
labels[150:160]=20
labels[160:172]=21
labels[172:183]=22
labels[183:195]=23
labels[195:205]=24
labels[205:217]=25
labels[217:227]=26
labels[227:240]=27
labels[240:252]=28
labels[252:262]=2
labels[262:274]=29
labels[274:285]=30
labels[285:296]=31
labels[296:307]=32
labels[307:318]=33
labels[318:329]=34
labels[329:339]=35
labels[339:349]=36
labels[349:359]=37
labels[359:369]=38
labels[369:377]=3
labels[377:388]=39
labels[388:400]=4
labels[400:408]=5
labels[408:418]=6
labels[418:429]=7
labels[429:443]=8
	  
Y = np_utils.to_categorical(labels, num_classes)
x,y = shuffle(img_data,Y, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
X_train = X_train.reshape(X_train.shape[0], img_cols, img_rows, 1)
X_test = X_test.reshape(X_test.shape[0], img_cols, img_rows, 1)
input_shape=(img_cols, img_rows, 1)

input_shape=img_data[0].shape
					
model = Sequential()

model.add(Convolution2D(16,(5,5),border_mode='same',input_shape=input_shape))
model.add(Activation('relu'))

model.add(Convolution2D(32, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Convolution2D(32, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))



model.add(Flatten())
model.add(Dense(300))
model.add(Activation('relu'))

model.add(Dropout(0.5))


model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='RMSProp',metrics=["accuracy"])

hist = model.fit(X_train, y_train, batch_size=32, nb_epoch=num_epoch, verbose=1, validation_data=(X_test, y_test))

from keras import callbacks

filename='model_train_new.csv'
csv_log=callbacks.CSVLogger(filename, separator=',', append=False)

early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='min')

filepath="Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"

checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')

callbacks_list = [csv_log,early_stopping,checkpoint]

hist = model.fit(X_train, y_train, batch_size=16, nb_epoch=num_epoch, verbose=1, validation_data=(X_test, y_test),callbacks=callbacks_list)


score = model.evaluate(X_test, y_test,  verbose=0)
score1 = model.evaluate(X_train, y_train,  verbose=0)


print('Test accuracy:', score[1])
print('Train accuracy:', score1[1])


test_image = cv2.imread('data/11. Acer palmaturu/download.jpg')
test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image=cv2.resize(test_image,(128,128))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print (test_image.shape)
   
if num_channel==1:
	if K.image_dim_ordering()=='th':
		test_image= np.expand_dims(test_image, axis=0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=3) 
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
		
else:
	if K.image_dim_ordering()=='th':
		test_image=np.rollaxis(test_image,2,0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
		
print((model.predict(test_image)))
print(model.predict_classes(test_image))

from sklearn.metrics import confusion_matrix
import itertools

Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)
print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))

from keras.models import model_from_json
from keras.models import load_model

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("leaf_model_small.h5")
print("Saved model to disk")

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights(".h5")


model.save('leaf_model.hdf5')
loaded_model=load_model('leaf_model.hdf5')
