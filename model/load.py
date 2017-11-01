import numpy as np
import keras.models
from keras.models import model_from_json, load_model
from scipy.misc import imread, imresize,imshow
import tensorflow as tf

def init(): 
	loaded_model = load_model('leaf_model_small.h5')
	loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

	graph = tf.get_default_graph()

	return loaded_model,graph
