from flask import Flask, render_template, request
import sys
import os,cv2
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras import backend as K
K.set_image_dim_ordering('tf')

from keras.utils import np_utils
from keras.models import load_model
from scipy.misc import imsave, imread, imresize
import keras.models
import re

sys.path.append(os.path.abspath("./model"))
from load import * 
app = Flask(__name__)
global model, graph
model, graph = init()

app = Flask(__name__)

num_channel = 1
P = os.getcwd() + r'//test_data'

@app.route('/upload')
def uploaded_file():
	return render_template('upload.html')

#@app.route('/')

@app.route('/uploader',methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f_name = f.filename
        PATH = P + r'//' + f_name  
        test_image = cv2.imread(PATH)
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
                                         
                    # Predicting the test image
        with graph.as_default():
            out = model.predict(test_image)
            
            print(out)
            print(np.argmax(out,axis=1))
            l = np.array_str(np.argmax(out,axis=1))
            x = ['[',']']
            p =[]
            for i in l:
                    if i not in x:
                            p.append(i)
            m = ''.join(p)
            m=int(m)                
                
                              
            data_dict = {
                    
                2:['Populus nigra',str(max(max(out))*100)],
                8:['Betula pubescens',str(max(max(out))*100)],
                18:['Polypodium vulgare',str(max(max(out))*100)],
                37:['Chelidonium majus',str(max(max(out))*100)],
                39:['Fragaria vesca',str(max(max(out))*100)]
                }            
            r = data_dict[m]
            r = ' '.join(r)                            
            
            return render_template('final.html',r=r, name = f_name )






if __name__ == "__main__":

    app.run(debug = True)
