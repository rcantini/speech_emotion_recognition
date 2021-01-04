### experiment reproducibility ###
seed_value= 42
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.random.set_seed(seed_value)


import numpy as np
import keras
from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine.saving import model_from_json
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils.vis_utils import plot_model
import seaborn
import EmotionRecognition.preprocessing as pre_proc
import matplotlib.pyplot as plt


### enable/disable attention ###
ENABLE_ATTENTION = True


def create_model(units=256):
    input = keras.Input(shape=(pre_proc.N_FRAMES, pre_proc.N_FEATURES))
    if MODEL == "Attention_BLSTM":
        states, forward_h, _, backward_h, _ = layers.Bidirectional(
            layers.LSTM(units, return_sequences=True, return_state=True)
        )(input)
        last_state = layers.Concatenate()([forward_h, backward_h])
        hidden = layers.Dense(units, activation="tanh", use_bias=False,
                              kernel_initializer=keras.initializers.RandomNormal(mean=0., stddev=1.)
                              )(states)
        out = layers.Dense(1, activation='linear', use_bias=False,
                              kernel_initializer=keras.initializers.RandomNormal(mean=0., stddev=1.)
                              )(hidden)
        flat = layers.Flatten()(out)
        energy = layers.Lambda(lambda x:x/np.sqrt(units))(flat)
        normalize = layers.Softmax()
        normalize._init_set_name("alpha")
        alpha = normalize(energy)
        context_vector = layers.Dot(axes=1)([states, alpha])
        context_vector = layers.Concatenate()([context_vector, last_state])
    elif MODEL == "BLSTM":
        context_vector = layers.Bidirectional(layers.LSTM(units, return_sequences=False))(input)
    else:
        raise Exception("Unknown model architecture!")
    pred = layers.Dense(pre_proc.N_EMOTIONS, activation="softmax")(context_vector)
    model = keras.Model(inputs=[input], outputs=[pred])
    model._init_set_name(MODEL)
    print(str(model.summary()))
    return model


def train_and_test_model(model):
    X_train, X_test, y_train, y_test = pre_proc.get_train_test()
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    plot_model(model, MODEL+"_model.png", show_shapes=True)
    best_weights_file = MODEL+"_weights.h5"
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=10)
    mc = ModelCheckpoint(best_weights_file, monitor='val_loss', mode='min', verbose=2,
                         save_best_only=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=32,
        callbacks=[es, mc],
        verbose=2
    )
    save(model)
    # model testing
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    if MODEL == "Attention_BLSTM":
        plt.title('model accuracy - BLSTM with attention')
    else:
        plt.title('model accuracy - BLSTM without attention')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(MODEL+"_accuracy.png")
    plt.gcf().clear()  # clear
    # loss on validation
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    if MODEL == "Attention_BLSTM":
        plt.title('model loss - BLSTM with attention')
    else:
        plt.title('model loss - BLSTM without attention')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(MODEL+"_loss.png")
    plt.gcf().clear()  # clear
    # test acc and loss
    model.load_weights(best_weights_file) # load the best saved model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    test_metrics = model.evaluate(X_test, y_test, batch_size=32)
    print("\n%s: %.2f%%" % ("test " + model.metrics_names[1], test_metrics[1] * 100))
    print("%s: %.2f" % ("test " + model.metrics_names[0], test_metrics[0]))
    print("test accuracy: " + str(format(test_metrics[1], '.3f')) + "\n")
    print("test loss: " + str(format(test_metrics[0], '.3f')) + "\n")
    # test acc and loss per class
    real_class = np.argmax(y_test, axis=1)
    pred_class_probs = model.predict(X_test)
    pred_class = np.argmax(pred_class_probs, axis=1)
    report = classification_report(real_class, pred_class)
    print("classification report:\n" + str(report) + "\n")
    cm = confusion_matrix(real_class, pred_class)
    print("confusion_matrix:\n" + str(cm) + "\n")
    data = np.array([value for value in cm.flatten()]).reshape(7,7)
    if MODEL == "Attention_BLSTM":
        plt.title('BLSTM with attention')
    else:
        plt.title('BLSTM without attention')
    seaborn.heatmap(cm, xticklabels=pre_proc.emo_labels_ita, yticklabels=pre_proc.emo_labels_ita, annot=data, cmap="Reds")
    plt.savefig(MODEL+"_conf_matrix.png")


def visualize_attention(model):
    best_weights_file = MODEL + "_weights.h5"
    model.load_weights(best_weights_file)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    _, X_test, _, _ = pre_proc.get_train_test()
    predictions = model.predict(X_test)
    labels = np.argmax(predictions, axis=1)
    # inspect attention weigths
    attention = model.get_layer(name="alpha")
    weigth_model = keras.Model(inputs=model.input, outputs=attention.output)
    attention_weights = weigth_model.predict(X_test)
    d = {}
    for w, l in zip(attention_weights, labels):
        if l not in d:
            d[l] = w
        else:
            d[l] += w
    data = []
    for x, y in d.items():
        norm_w = y / np.sum(y)
        data.append(norm_w)
    #reshape and trim
    bins = 10
    bin_c = pre_proc.N_FRAMES//bins
    trim = pre_proc.N_FRAMES%bins
    data = np.asarray(data).reshape(pre_proc.N_EMOTIONS, pre_proc.N_FRAMES)[:, trim:]
    data = np.sum(data.reshape([7, bins, bin_c]), axis=2).reshape(pre_proc.N_EMOTIONS,bins)
    plt.clf()
    seaborn.heatmap(data, yticklabels=pre_proc.emo_labels_ita, cmap="Reds")
    plt.savefig("visualize_attention.png")


def load():
    with open("model.json", 'r') as f:
        model = model_from_json(f.read())
    best_weights_file = MODEL + "_weights.h5"
    # Load weights into the new model
    model.load_weights(best_weights_file)
    return model


def save(model):
    model_json = model.to_json()
    with open(MODEL+"_model.json", "w") as json_file:
        json_file.write(model_json)
    print("model saved")




######### SPEECH EMOTION RECOGNITION #########

# 1) feature extraction
pre_proc.feature_extraction()

# 2) select model
if ENABLE_ATTENTION:
    MODEL = "Attention_BLSTM"
else:
    MODEL = "BLSTM"

# 3) create model
model = create_model()

# 4) train and test model
train_and_test_model(model)

# 5) visualize attention weights
if ENABLE_ATTENTION:
    visualize_attention(model)