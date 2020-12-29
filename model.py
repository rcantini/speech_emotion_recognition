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
        tanh = layers.TimeDistributed(layers.Activation("tanh"))(states)
        scores = layers.TimeDistributed(layers.Dense(1, activation='linear', use_bias=False))(tanh)
        f_scores = layers.Flatten()(scores)
        a = layers.Softmax()(f_scores)
        r = layers.Dot(axes=1)([states, a])
        vec = layers.Concatenate()([r, last_state])
    elif MODEL == "BLSTM":
        vec = layers.Bidirectional(layers.LSTM(units, return_sequences=False))(input)
    else:
        raise Exception("Unknown model architecture!")
    pred = layers.Dense(pre_proc.N_EMOTIONS, activation="softmax")(vec)
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
    seaborn.heatmap(cm, xticklabels=pre_proc.emo_labels_ita, yticklabels=pre_proc.emo_labels_ita, annot=data, cmap="Reds")
    plt.savefig(MODEL+"_conf_matrix.png")


def load():
    with open("model.json", 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    model.load_weights("weights.h5")
    return model


def save(model):
    model_json = model.to_json()
    with open(MODEL+"_model.json", "w") as json_file:
        json_file.write(model_json)
    print("model saved")
    return None


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
