import math
import librosa
import os
import matplotlib.pyplot as plt
from keras_preprocessing.sequence import pad_sequences
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as std, OneHotEncoder as enc
from imblearn.over_sampling import SMOTE
import pickle


# constants
np.random.seed(42)
sr = 16000
duration = 5
frame_length = 512
N_FRAMES = math.ceil(sr*duration/frame_length)
N_FEATURES = 46
N_EMOTIONS = 7
emo_codes = {"W": 0, "L": 1, "E": 2, "A": 3, "F": 4, "T": 5, "N": 6}
emo_labels_deu = ["wut", "langeweile", "ekel", "angst", "freude", "trauer", "neutral"]
emo_labels_en = ["anger", "boredom", "disgust", "fear", "happiness", "sadness", "neutral"]
emo_labels_ita = ["rabbia", "noia", "disgusto", "paura", "felicità", "tristrezza", "neutro"]
path = "emo_db/wav/"


def get_emotion_label(file_name):
    emo_code = file_name[5]
    return emo_codes[emo_code]


def get_emotion_name(file_name, lang="ita"):
    emo_code = file_name[5]
    if lang == "deu":
        return emo_labels_deu[emo_codes[emo_code]]
    elif lang == "en":
        return emo_labels_en[emo_codes[emo_code]]
    elif lang == "ita":
        return emo_labels_ita[emo_codes[emo_code]]
    else:
        raise Exception("wrong language")


def get_speech_text(file_name):
    utt_code = file_name[3:6]
    if utt_code == "a01":
        return "\"Der Lappen liegt auf dem Eisschrank\"\n(The tablecloth is lying on the frigde)"
    if utt_code == "a02":
        return "\"Das will sie am Mittwoch abgeben\"\n(She will hand it in on Wednesday)"
    if utt_code == "a04":
        return "\"Heute abend könnte ich es ihm sagen\"\n(Tonight I could tell him)"
    if utt_code == "a05":
        return "\"Das schwarze Stück Papier befindet sich da oben neben dem Holzstück\"\n(The black sheet of paper is located up there besides the piece of timber)"
    if utt_code == "a07":
        return "\"In sieben Stunden wird es soweit sein\"\n(In seven hours it will be)"
    if utt_code == "b01":
        return "\"Was sind denn das für Tüten, die da unter dem Tisch stehen?\"\n(What about the bags standing there under the table?)"
    if utt_code == "b02":
        return "\"Sie haben es gerade hochgetragen und jetzt gehen sie wieder runter\"\n(They just carried it upstairs and now they are going down again)"
    if utt_code == "b03":
        return "\"An den Wochenenden bin ich jetzt immer nach Hause gefahren und habe Agnes besucht\"\n(Currently at the weekends I always went home and saw Agnes)"
    if utt_code == "b09":
        return "\"Ich will das eben wegbringen und dann mit Karl was trinken gehen\"\n(I will just discard this and then go for a drink with Karl)"
    if utt_code == "b10":
        return "\"Die wird auf dem Platz sein, wo wir sie immer hinlegen\"\n(It will be in the place where we always store it)"


def feature_extraction():
    wavs = []
    # load 16 kHz resampled files
    for file in os.listdir(path):
        y, _ = librosa.load(path + "/" + file, sr=sr, mono=True, duration=duration)
        wavs.append(y)
    # pad to fixed length (zero, 'pre')
    wavs_padded = pad_sequences(wavs, maxlen=sr * duration, dtype="float32")
    features = [] #(N_SAMPLES, N_FRAMES, N_FEATURES)
    emotions = []
    for y, name in zip(wavs_padded, os.listdir(path)):
        frames = []
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=frame_length)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=frame_length)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=frame_length)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=frame_length)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=frame_length)[0]
        S, phase = librosa.magphase(librosa.stft(y=y, hop_length=frame_length))
        rms = librosa.feature.rms(y=y, hop_length=frame_length, S=S)[0]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=frame_length)
        mfcc_der = librosa.feature.delta(mfcc)
        for i in range(N_FRAMES):
            f=[]
            f.append(spectral_centroid[i])
            f.append(spectral_contrast[i])
            f.append(spectral_bandwidth[i])
            f.append(spectral_rolloff[i])
            f.append(zero_crossing_rate[i])
            f.append(rms[i])
            for m_coeff in mfcc[:,i]:
                f.append(m_coeff)
            for m_coeff_der in mfcc_der[:, i]:
                f.append(m_coeff_der)
            frames.append(f)
        features.append(frames)
        emotions.append(get_emotion_label(name))
    features = np.array(features)
    emotions = np.array(emotions)
    print(str(features.shape))
    pickle.dump(features, open("features.p", "wb"))
    pickle.dump(emotions, open("emotions.p", "wb"))


def get_train_test(test_samples_per_emotion=20):
    features = pickle.load(open("features.p", "rb"))
    emotions = pickle.load(open("emotions.p", "rb"))
    analyze_emotion_distribution(emotions, "original")
    # flatten
    N_SAMPLES = len(features)
    features.shape = (N_SAMPLES, N_FRAMES * N_FEATURES)
    # standardize data
    scaler = std()
    features = scaler.fit_transform(features)
    # shuffle
    perm = np.random.permutation(N_SAMPLES)
    features = features[perm]
    emotions = emotions[perm]
    # get balanced test set of real samples
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    count_test = np.zeros(N_EMOTIONS)
    for f,e in zip(features, emotions):
        if count_test[e] < test_samples_per_emotion:
            X_test.append(f)
            y_test.append(e)
            count_test[e]+=1
        else:
            X_train.append(f)
            y_train.append(e)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    # balance train classes
    sm = SMOTE()
    X_train, y_train = sm.fit_sample(X_train, y_train)
    # analyze emotion distribution
    analyze_emotion_distribution(y_test, "balanced_test_set")
    analyze_emotion_distribution(y_train, "balanced_SMOTE_training_set")
    # restore 3D shape
    X_train.shape = (len(X_train), N_FRAMES, N_FEATURES)
    X_test.shape = (len(X_test), N_FRAMES, N_FEATURES)
    # encode labels in one-hot vectors
    encoder = enc(sparse=False)
    y_train = np.array(y_train).reshape(-1, 1)
    y_train = encoder.fit_transform(y_train)
    y_test = np.array(y_test).reshape(-1, 1)
    y_test = encoder.fit_transform(y_test)
    return X_train, X_test, y_train, y_test


def analyze_emotion_distribution(emotions_list, descr = ""):
    d = {}
    for e in emotions_list:
        if e not in d:
            d[e] = 0
        d[e] += 1
    keys = []
    values = []
    for x, y in d.items():
        keys.append(emo_labels_ita[x])
        values.append(y)
    plt.bar(keys, values)
    plt.title("emotion distribution (total wavs: " + str(len(emotions_list)) + ")")
    plt.ylabel('number of samples')
    plt.savefig("emotion_distribution_"+descr+".png")
    plt.gcf().clear()
