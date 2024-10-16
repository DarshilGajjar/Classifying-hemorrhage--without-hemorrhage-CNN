from keras.preprocessing.image import ImageDataGenerator

def preprocess_data(train_data, test_data):
    Generator = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.25,
        shear_range=0.25,
        rotation_range=25,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.15
    )

    Train_IMG_Set = Generator.flow_from_dataframe(
        dataframe=train_data,
        x_col="JPG",
        y_col="CATEGORY",
        color_mode="grayscale",
        class_mode="categorical",
        subset="training"
    )

    Validation_IMG_Set = Generator.flow_from_dataframe(
        dataframe=train_data,
        x_col="JPG",
        y_col="CATEGORY",
        color_mode="grayscale",
        class_mode="categorical",
        subset="validation"
    )

    Test_IMG_Set = Generator.flow_from_dataframe(
        dataframe=test_data,
        x_col="JPG",
        y_col="CATEGORY",
        color_mode="grayscale",
        class_mode="categorical"
    )
    
    return Train_IMG_Set, Validation_IMG_Set, Test_IMG_Set