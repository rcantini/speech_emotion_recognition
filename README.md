# speech_emotion_recognition

How to detect emotions from speech using **Bi-directional LSTM** networks and **attention** mechanism in Keras

The model is aimed at classifying utterances from the **Berlin Dataset of Emotional Speech** (EMO-DB), according to the expressed emotion.
Considered emotions are: anger, boredom, disgust, fear, happiness, sadness and neutral.

The model is composed by the following steps:
- *Feature extraction*: features are extracted by exploiting the librosa library. Considered features are: spectral_centroid, spectral_contrast, spectral_bandwidth, spectral_rolloff, zero_crossing_rate, rms, mfcc and mfcc's first order derivatives.
- *Model training*: For this task I used a bi-directional LSTM network enhanced with attention.
- *Performance evaluation*: I evaluated the trained model using 20 samples per emotion, achieving a 90\% accuracy. In order to assess the benefits brought by the attention mechanism, I tested a simplified version of the model in which attention is absent which achieved about 77\% accuracy.
