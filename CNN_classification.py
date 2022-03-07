import keras
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np


def model_split(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1, shuffle=Y)

    X_train, X_validate, Y_train, Y_validate = train_test_split(X_train, Y_train, test_size=0.25, random_state=1,
                                                                shuffle=Y)
    return X_train, X_test, X_validate, Y_train, Y_test, Y_validate


def model_classifier(name_model,n_classes, X, Y, model, batch_size, epochs, early_stop):
    """
    :param n_classes:
    :param name_model:
    :param X: nparray
    :param Y: nparray
    :param model: CNN_models
    :param batch_size: int
    :param epochs: int
    :param early_stop: int
    :return:
    """
    kernels, chans, samples = 1, X.shape[1], X.shape[2]
    X_train, X_test, X_validate, Y_train, Y_test, Y_validate = model_split(X, Y)
    if name_model == "ATCNet":
        X_train = X_train.reshape(X_train.shape[0], kernels, chans, samples)
        X_validate = X_validate.reshape(X_validate.shape[0], kernels, chans, samples)
        X_test = X_test.reshape(X_test.shape[0], kernels, chans, samples)
    else:
        X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
        X_validate = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
        X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)

    # compile the model and set the optimizers
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    # set a valid path for your system to record model checkpoints
    checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=3,
                                   save_best_only=True)

    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stop)  # 100
    ##########################################################################
    # the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
    # the weights all to be 1
    if n_classes == 2:
        class_weights = {0: 1, 1: 1}
    elif n_classes == 3:
        class_weights = {0: 1, 1: 1, 2: 1}
    else:
        raise

    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
                        verbose=1, validation_data=(X_validate, Y_validate),
                        callbacks=[checkpointer, callback], class_weight=class_weights)

    # load optimal weights
    model.load_weights('/tmp/checkpoint.h5')
    probs = model.predict(X_test)
    Y_pred = probs.argmax(axis=-1)
    scores = model_score_values(X_train, X_test, X_validate, Y_train, Y_test, Y_validate, model)
    return model, history, scores, Y_test, Y_pred


def model_score_values(X_train, X_test, X_validate, Y_train, Y_test, Y_validate, model_CNN):
    scores = {}
    validation_score = model_CNN.evaluate(X_validate, Y_validate)
    scores["validation"] = validation_score
    train_score = model_CNN.evaluate(X_train, Y_train)
    scores["train"] = train_score
    test_score = model_CNN.evaluate(X_test, Y_test)
    scores["test"] = test_score

    return scores
