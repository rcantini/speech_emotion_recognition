# speech_emotion_recognition

How to detect emotions from speech using **Bi-directional LSTM** networks and **attention** mechanism in Keras

The model is aimed at classifying utterances from the **Berlin Dataset of Emotional Speech** (EMO-DB), according to the expressed emotion.
Considered emotions are: *anger*, *boredom*, *disgust*, *fear*, *happiness*, *sadness* and *neutral*.

The developed application is composed by the following steps:
- ***Feature extraction***: features are extracted by exploiting **Librosa**, a python package for music and audio analysis. Considered features are: *spectral centroid*, *spectral contrast*, *spectral bandwidth*, *spectral rolloff*, *zero crossing rate*, *rms*, *mfcc* and mfcc's *first order derivatives*.
- ***Class balancing***: I used **SMOTE** for dealing with class imbalance.
- ***Model training***: I trained a bi-directional LSTM network enhanced with attention.
- ***Performance evaluation***: I evaluated the trained model using 20 samples per emotion, achieving 90\% accuracy. In order to assess the benefits brought by the attention mechanism, I tested a simplified version of the model in which attention is absent which achieved about 77\% accuracy.
