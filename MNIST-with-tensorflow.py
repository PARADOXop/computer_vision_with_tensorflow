'''C:/Users/rkuka/miniconda3/Scripts/activate
conda activate tf2minilearn
To check the list of devices with tensorflow:
tensorflow.config.list_physical_devices()

1. To check all the venvs 
  conda info --envs 

2. How to create new venvs
	conda create -n NAME_OF_VIRTUAL_ENV python=PYTHON_VERSION

3. To activate venv use 
	conda activate NAME_OF_VIRTUAL_ENV

4. To deactivate venv use 
	conda deactivate NAME_OF_VIRTUAL_ENV'''

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D
from show import display_some_examples
from models  import MyCustomModel, Functinal_model


#************************************ WAYS TO CREATE MODEL*************************************

#1 tensorflow.keras.Sequential
'''Not a good way to build a model beacause of less flexibility with parameters'''
model1 = tf.keras.Sequential(
    [
        Input(shape = (28,28, 1 )),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(), 

        Conv2D(128, (3, 3), activation = 'relu'),
        MaxPool2D(),
        BatchNormalization(),
     
        GlobalAvgPool2D(),
        Dense(64, activation = 'relu'),
        Dense(10, activation = 'softmax')     
    ])

#*******************************************************************************************************************************************



if __name__ == '__main__':
    tf.config.list_physical_devices()
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    print("X_train", X_train.shape)
    print("y_train", y_train.shape)

    print("X_test", X_test.shape)
    print("y_test", y_test.shape)
#*******************************************Displaying images********************************************   
    n = input("want to display some Images\n"
                  "y for yes\n"
                  "0 for no")
    print("**************************************************************************************")
    if n == 'y':
        display_some_examples(X_test, y_test)
     
        
#***********************************TRAINING THE MODEL******************************
    
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    
    '''this is what we can as normalization in with we try to reduce the range of 
       this is because X_train has an array with consist int value (0, 255) and if we divide values by 255 there will be no float value so we change the data type to float..   
    '''
# our input has size of (28,28), but our model exepts Inputs in (28, 28, 3) format So lets change'em
   
    X_train = np.expand_dims(X_train, axis = -1)
    X_test = np.expand_dims(X_test, axis = -1)
    
    '''axis is used to exapnd the dimension as our input for train data was (60, 28, 28) with is 3 dimension
    0 to 2 so we can write axis as -1 or 3 to expand it for 1 dimension'''
    
    # check the shape of train and test inputs
    
    '''print("X_train", X_train.shape)
    print("y_train", y_train.shape)

    print("X_test", X_test.shape)   
    print("y_test", y_test.shape)'''
    
    
    print("which model you want to use")
    q = int(input("1 for Sequential model\n"
                  "2 for Functinal_model\n"
                  "3 for inherits from this class\n"
                  "4 for exit"))
    
#***********************************************************METHOD NUMBER 1*****************************************************************
    if q == 1:
        print("*********************************************************************************************")
        print("USING tensorflow.keras.Sequential as model")
        print("*********************************************************************************************")
        model1.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
    #training
        model1.fit(X_train, y_train, batch_size = 64, epochs = 3, validation_split = 0.2)
    #evaluation on test set
        model1.evaluate(X_test, y_test, batch_size = 32)
        
        
        
#*******************************************METHOD NUMBER 2*****************************************************************
    elif q == 2:
        model = Functinal_model()
        model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
    #training
        model.fit(X_train, y_train, batch_size = 64, epochs = 3, validation_split = 0.2)
    #evaluation on test set
        model.evaluate(X_test, y_test, batch_size = 32)
        
     
        
#*********************************************METHOD NUMBER 3*****************************************************************
    def custom_model():
        model = MyCustomModel()
        model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
    #training
        model.fit(X_train, y_train, batch_size = 64, epochs = 3, validation_split = 0.2)
    #evaluation on test set
        model.evaluate(X_test, y_test, batch_size = 32)
        
    if q == 3:
        custom_model()  
    
    elif q == 4:
        exit()
    