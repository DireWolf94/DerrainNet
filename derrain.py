import numpy as np
import keras
import h5py
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D,Conv2DTranspose,Add
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model,Sequential
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.layers.advanced_activations import PReLU

import matplotlib.pyplot as plt
from keras.preprocessing import image
from matplotlib.pyplot import imshow
import PIL
import scipy.misc

class Derain(object):

	path = ""

    def __init__(self, data_dir,checkpoint_dir='./checkpoints/'):

        self.X_train=[]
        self.Y_train=[]
        self.X_test=[]
        self.Y_test=[]
        self.model=keras.models.load_model("model.h5")

        self.train_dir=data_dir
        """
            data_directory : path like /home/kushagr/NNFL_Project/rain/training/
            	includes the dataset folder with '/'
            Initialize all your variables here
        """

    def test(self,test_data_dir,no_of_test_images):

        

        for i in range(no_of_test_images):
            test_img_path = test_data_dir+str(i+1)+".jpg"
            test_img = image.load_img(test_img_path, target_size=(128, 256))
            test_x = image.img_to_array(test_img)
            test_Y=test_x[:,:128,:]
            test_X=test_x[:,128:,:]
            self.X_test.append(test_X)
            self.Y_test.append(test_Y)

        self.X_test=np.array(self.X_test)
        self.Y_test=np.array(self.Y_test)    
        self.X_test=self.X_test/255.
        self.Y_test=self.Y_test/255.

        print(self.model.evaluate(x=self.X_test,y=self.Y_test,batch_size=1))

        
    def train(self, training_steps=10):

    	for i in range(700):
            img_path = self.train_dir+str(i+1)+".jpg"
            img = image.load_img(img_path, target_size=(128, 256))
            x = image.img_to_array(img)
            Y=x[:,:128,:]
            X=x[:,128:,:]
            self.X_train.append(X)
            self.Y_train.append(Y)

        self.X_train=np.array(self.X_train)
        self.Y_train=np.array(self.Y_train)    
        self.X_train=self.X_train/255.
        self.Y_train=self.Y_train/255.

        inputs = Input((128, 128, 3))
        conv1 = Conv2D(64, (3, 3),  padding='same')(inputs)
        act1=PReLU()(conv1)
        batch1=BatchNormalization(axis=3)(act1) 
        pool1 = MaxPooling2D(pool_size=(2, 2),padding='same')(batch1)
        pool1 = ZeroPadding2D((32, 32))(pool1)
        print(pool1.shape)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        act2=PReLU()(conv2)
        batch2=BatchNormalization(axis=3)(act2)
        pool2 = MaxPooling2D(pool_size=(2, 2),padding='same')(batch2)
        pool2 = ZeroPadding2D((32, 32))(pool2)
        print(pool2.shape)

        conv3 = Conv2D(64, (3, 3),  padding='same')(pool2)
        act3=PReLU()(conv3)
        batch3=BatchNormalization(axis=3)(act3)
        pool3 = MaxPooling2D(pool_size=(2, 2),padding='same')(batch3)
        pool3 = ZeroPadding2D((32, 32))(pool3)
        print(pool3.shape)


        conv4 = Conv2D(64, (3, 3), padding='same')(pool3)
        act4=PReLU()(conv4)
        batch4=BatchNormalization(axis=3)(act4)
        pool4 = MaxPooling2D(pool_size=(2, 2),padding='same')(batch4)
        pool4 = ZeroPadding2D((32, 32))(pool4)
        print(pool4.shape)

        conv5 = Conv2D(32, (3, 3), padding='same')(pool4)
        act5=PReLU()(conv5)
        batch5=BatchNormalization(axis=3)(act5)
        pool5 = MaxPooling2D(pool_size=(2, 2),padding='same')(batch5)
        pool5 = ZeroPadding2D((32, 32))(pool5)
        print(pool5.shape)

        conv6 = Conv2D(1, (3, 3), activation='relu', padding='same')(pool5)
        act6=PReLU()(conv6)
        batch6=BatchNormalization(axis=3)(act6)
        pool6 = MaxPooling2D(pool_size=(2, 2),padding='same')(batch6)
        pool6 = ZeroPadding2D((32, 32))(pool6)
        print(pool6.shape)

        deconv1 = Conv2DTranspose(32, (3, 3), padding='same')(pool6)
        print(deconv1.shape)
        dbatch1=BatchNormalization(axis=3)(deconv1) 
        dact1 = Activation('relu')(dbatch1)

        deconv2 = Conv2DTranspose(64, (3, 3), padding='same')(dact1)
        print(deconv2.shape)
        dbatch2=BatchNormalization(axis=3)(deconv2) 
        dact2 = Activation('relu')(dbatch2)

        m1 = Add()([pool4, dact2])

        deconv3 = Conv2DTranspose(64, (3, 3), padding='same')(m1)
        dbatch3=BatchNormalization(axis=3)(deconv3) 
        dact3 = Activation('relu')(dbatch3)

        deconv4 = Conv2DTranspose(64, (3, 3), padding='same')(dact3)
        dbatch4=BatchNormalization(axis=3)(deconv4) 
        dact4 = Activation('relu')(dbatch4)

        m2 = Add()([pool2, dact4])

        deconv5 = Conv2DTranspose(64, (3, 3), padding='same')(m2)
        dbatch5=BatchNormalization(axis=3)(deconv5) 
        dact5 = Activation('relu')(dbatch5)

        deconv6 = Conv2DTranspose(3, (3, 3), padding='same')(dact5)
        dbatch6=BatchNormalization(axis=3)(deconv6) 
        dact6 = Activation('relu')(dbatch6)

        m3 = Add()([inputs, dact6])
        out=Activation('tanh')(m3)

        self.model = Model(inputs=[inputs], outputs=[out])

        rms = keras.optimizers.RMSprop(lr=0.1, rho=0.9, epsilon=1e-08, decay=0.000)

        self.model.compile(optimizer=rms, loss='mean_squared_error', metrics=['accuracy'])
   
        self.model.fit(x = self.X_train, y = self.Y_train, epochs =training_steps ,batch_size=1 )
        """
            Trains the model on data given in path/train.csv
            	which conatins the RGB values of each pixel of the image  

            No return expected
        """
        

    def save_model(self, step):

        self.model.save("model.h5")
        """
            saves model on the disk
            You can use pickle or Session.save in TensorFlow
            no return expected
        """


    def load_model(self, **params):

        self.model=keras.models.load_model("model.h5")
    	# file_name = params['name']
        # return pickle.load(gzip.open(file_name, 'rb'))

        """
            returns a pre-trained instance of Segment class
        """
        return self.model

