
'''Return a possibly-empty list of path names that match pathname, which must be a string 
    containing a path specification. pathname can be either absolute (like /usr/src/Python-1.5/Makefile) 
    or relative (like ../../Tools/*/*.gif), and can contain shell-style wildcards. Broken symlinks are 
    included in the results (as in the shell). Whether or not the results are sorted depends on the file 
    system. If a file that satisfies conditions is removed or added during the call of this function, whether 
    a path name for that file be included is unspecified.'''
    
    
import os
import glob
import csv
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def order_test_set(path_to_images, path_to_csv):
    
    try:
        with open(path_to_csv, 'r') as csvfile:

            reader = csv.reader(csvfile, delimiter=',')

            for i, row in enumerate(reader):

                if i==0:
                    continue

                img_name = row[-1].replace('Test/', '')
                label = row[-2]

                path_to_folder = os.path.join(path_to_images, label)

                if not os.path.isdir(path_to_folder):
                    os.makedirs(path_to_folder)

                img_full_path = os.path.join(path_to_images, img_name)
                shutil.move(img_full_path, path_to_folder)

    except:
        print('[INFO] : Error reading csv file')

def split_data(path_to_data, path_to_save_train, path_to_save_val, split_size = 0.2):
    folders = os.listdir(path_to_data)   
    for folder in folders:
        full_path = os.path.join(path_to_data, folder)
        images_path = glob.glob(os.path.join(full_path, '*.png'))
        
        X_train, X_val = train_test_split(images_path, test_size = split_size)
        
        for x in X_train:
            
            
            path_to_folder = os.path.join(path_to_save_train, folder)
            
            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)
                
            shutil.copy(x, path_to_folder)
            
        for x in X_val:
            
            
            path_to_folder = os.path.join(path_to_save_val, folder)
            
            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)
                
            shutil.move(x, path_to_folder)
            
            
def create_generators(batch_size, train_data_path, val_data_path, test_data_path):
    
    train_preprocessor = ImageDataGenerator(
        rescale = 1 / 255.,
        rotation_range=10,
        width_shift_range=0.1
    )

    test_preprocessor = ImageDataGenerator(
        rescale = 1 / 255.,
    )

    train_generator = train_preprocessor.flow_from_directory(
        train_data_path,
        class_mode="categorical",
        target_size=(60,60),
        color_mode='rgb',
        shuffle=True,
        batch_size=batch_size
    )

    val_generator = test_preprocessor.flow_from_directory(
        val_data_path,
        class_mode="categorical",
        target_size=(60,60),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size,
    )

    test_generator = test_preprocessor.flow_from_directory(
        test_data_path,
        class_mode="categorical",
        target_size=(60,60),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size,
    )

    return train_generator, val_generator, test_generator