import os
import my_predictor
import opendatasets as od
from models import streetsigns_model
from utils import split_data, order_test_set, create_generators
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
od.download('https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign')
            
if __name__=='__main__':
    
    path_to_data = './gtsrb-german-traffic-sign/Train'
    path_to_main = './gtsrb-german-traffic-sign/training'
    path_to_save_train = './gtsrb-german-traffic-sign/training/train'
    path_to_save_val = './gtsrb-german-traffic-sign/training/validation'
    path_to_images = './gtsrb-german-traffic-sign/Test'
    path_to_csv = './gtsrb-german-traffic-sign/Test.csv'
    path_to_train = "./gtsrb-german-traffic-sign/training/train"
    path_to_val = "./gtsrb-german-traffic-sign/training/validation"
    path_to_test = "./gtsrb-german-traffic-sign/Test"
    
    
    if False:
        split_data(path_to_data, path_to_save_train, path_to_save_val)   
        order_test_set(path_to_images, path_to_csv)
    
    
    batch_size = 64
    epochs = 15
    lr=0.0001

    train_generator, val_generator, test_generator = create_generators(batch_size, path_to_train, path_to_val, path_to_test)
    nbr_classes = train_generator.num_classes

    TRAIN=False
    TEST=True

    if TRAIN:
        path_to_save_model = 'P:\\Machine learning\\gtsrb-german-traffic-sign\\models'
        ckpt_saver = ModelCheckpoint(
            path_to_save_model,
            monitor="val_accuracy",
            mode='max',
            save_best_only=True,
            save_freq='epoch',
            verbose=1
        )

        early_stop = EarlyStopping(monitor="val_accuracy", patience=10)

        model = streetsigns_model(nbr_classes)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=True)
        
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(train_generator,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=val_generator,
                callbacks=[ckpt_saver, early_stop]
                )
        
    if TEST:
        model = tf.keras.models.load_model(path_to_save_model)
        model.summary()

        print("Evaluating validation set:")
        model.evaluate(val_generator)

        print("Evaluating test set : ")
        model.evaluate(test_generator)