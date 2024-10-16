# Evaluating the performance of CNN model
Model_Results = Model.evaluate(Test_IMG_Set, verbose=False)

# Predicting using the trained model
Prediction = Model.predict(Test_IMG_Set)
Prediction = Prediction.argmax(axis=-1)