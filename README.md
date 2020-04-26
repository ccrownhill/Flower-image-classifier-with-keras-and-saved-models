# Flower-image-classifier-with-keras-and-saved-models
3 keras models trained on flower dataset from tensorflow with script to use any of those models to predict some image.

The code for any of the trained model saves model1.h5, model2.h5 and model3.h5 is in the respective model{n}.py file which uses tensorflow version 2.1. In order to make a prediction on any .jpg image you need to use the predict.py file like this

python predict.py "image file" "saved model file"

So in order to make a prediction on the image "cat.jpg" using "model1.h5" you should run the following

python predict.py "rose.jpg" "model1.h5"

Here are the accuracies on the training data set of all models (if you just want to pick one you should take Model 5 as it is probably not too overfitted because of a dropout layer):

Model 1 --> between 85% and 90%

Model 2 --> 96%

Model 3 --> 96%
