import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# First we are going to build the CNN model
def build_cnn_model():
    Model = Sequential()

    Model.add(Conv2D(12,(3,3),activation="relu",input_shape=(256,256,1)))
    Model.add(BatchNormalization())
    Model.add(MaxPooling2D((2,2)))
    
    Model.add(Conv2D(24,(3,3),activation="relu",padding="same"))
    Model.add(Dropout(0.2))
    Model.add(MaxPooling2D((2,2)))
    
    Model.add(Conv2D(64,(3,3),activation="relu",padding="same"))
    Model.add(Dropout(0.5))
    Model.add(MaxPooling2D((2,2)))
    
    Model.add(Flatten())
    Model.add(Dense(256, activation="relu"))
    Model.add(Dropout(0.5))
    Model.add(Dense(2, activation="softmax"))

    return Model

def train_model(model, Train_IMG_Set, Validation_IMG_Set):
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    Call_Back = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5, mode="min")
    
    CNN_Model = model.fit(Train_IMG_Set, 
                    validation_data=Validation_IMG_Set, callbacks=Call_Back, 
                    batch_size=32, 
                    epochs=50)
    
    return CNN_Model

# Now we are going to train our CNN model on our dataset
# Early stopping callback
Call_Back = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=5, mode="min")

# Compiling the CNN model
Model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Training the CNN model
CNN_Model = Model.fit(Train_IMG_Set,
                      validation_data=Validation_IMG_Set,
                      callbacks=Call_Back,
                      batch_size=32,
                      epochs=50)
