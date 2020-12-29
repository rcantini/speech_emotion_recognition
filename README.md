# speech_emotion_recognition

How to build a dog breed classifier with Keras and Tensorflow using Convolutional Neural Networks.

The model is aimed at classifying dog images belonging to 2 particular breeds: Chihuahua and pug.

In order to cope with the small amount of traning data, the model exploits three main techniques:

Real time data augmentation during training
Transfer Learning (VGG16 CNN architecture, ImageNet)
Fine tuning
The obtained model achieves a classification accuracy of about 93% on test images.
