import numpy as np

np.random.seed(999)
import matplotlib.pyplot as pyplot
from keras.utils import to_categorical
from sklearn.preprocessing import scale
from keras.models import Sequential
from keras import metrics
from keras.optimizers import RMSprop
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.layers import Dense
from keras.layers import LSTM, Activation, Dropout
import pickle
from keras.optimizers import SGD
from keras.layers import Dense, Merge
from keras.layers.normalization import BatchNormalization
import math
import pandas as pd
import random
import json
import os
import uuid
from sklearn.metrics import confusion_matrix
import itertools
import data_pipelines as dp

TIMESERIES_FOLDER = "data/timeseries/"
NEWS_FOLDER = "data/news"


def get_formated_data(ticker, output_shape, train_test_val_ratio=[0.7, 0.2, 0.1], sliding_window=True,
                      classification=True):
    """method to do necessary data massage
    sliding window - if True, then input_dim[0] sliding windows is created;
    output_shape:
        defined the output shape of X samples;
        Y outputs is (output_shape[0], 1), where 1 - is exactly one value in the end of each time series.
    classification - defined if Y should contain boolean value (1 if stock grow, 0 if fell.
                    if classification = False, then stock price (float) is returned.
    """

    ts_file = TIMESERIES_FOLDER + "{0}_df.p".format(ticker)
    with open(ts_file, "rb") as ts_file:
        ticker_df = pickle.load(ts_file)

    news_file = NEWS_FOLDER + "{0}_df.p".format(ticker)
    with open(ts_file, "rb") as ts_file:
        news_df = pickle.load(news_file)



    if sliding_window:  # implementation of sliding window. Each window is moving by 1 element.
        if len(output_shape) != 3:
            raise ValueError("If you choose sliding window parameter"
                             "please specify 3 output dimensions for data."
                             "Now you passed {0}".format(len(output_shape)))

        # Explicitly define output dimensions
        batch_size = output_shape[0]
        timesteps = output_shape[1]
        features = output_shape[2]

        # Get X properly shaped
        X = dp.convert_data_to_batch_timesteps(ticker_df.loc[:, ['open', 'high', 'low', 'close', 'volume', 'date']],
                                            batch_size=batch_size, timesteps=timesteps, features=features)
        nb_samples = np.shape(X)[0]

        # Get Y properly shaped
        target_df = dp.convert_ts_to_categorial(ticker_df, timesteps=timesteps)

        target_df = target_df.loc[:(nb_samples - 1), :]  # shape Y to trimmed version of X

        if classification:
            Y = target_df.close_bool.as_matrix()
        else:
            Y = target_df.close.as_matrix()

    else:

        raise NotImplemented("Shaping time-series data without sliding windows are to be implemented if needed.")

        if len(output_shape) != 2:
            raise ValueError("Please specify output with shape (batch_size, feature_size)"
                             "Now you passed {0}".format(len(output_shape)))

        # Explicitly define output dimensions
        batch_size = output_shape[0]
        features = output_shape[1]

    # Define indices for training, testing and validation data sets which conform to LSTM data shape requirements.
    train_index = int((nb_samples * train_test_val_ratio[0]) / batch_size) * batch_size
    test_index = int((nb_samples * train_test_val_ratio[1]) / batch_size) * batch_size

    Y_train = Y[:train_index]
    Y_test = Y[train_index:(train_index + test_index)]
    Y_val = Y[(train_index + test_index):]

    X_train = X[:train_index, :, :]
    X_test = X[train_index:(train_index + test_index), :, :]
    X_val = X[(train_index + test_index):, :, :]

    print("Input data shape: \n X train: {0}, Y train: {1} \n X test: {2}, Y test: {3} \n X val: {4}, Y val: {5}".
          format(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape, X_val.shape, Y_val.shape))

    return X_train, Y_train, X_test, Y_test, X_val, Y_val # TODO: this method should also return a prepared data for news.



def get_complex_model(mode):
    """a model which combine several branches with various inputs"""

    # A dictionary with specific config. Will be used to track experiments settings.
    model_config = {'name': 'complex_model'}
    model_config['optimizer'] = 'adam'

    if mode == 'regression':
        model_config['loss'] = 'mean_squared_error'
        model_config['metrics'] = ['mae']
    elif mode == 'classification':
        model_config['loss'] = 'binary_crossentropy'
        model_config['metrics'] = ['binary_accuracy']
    else:
        raise ValueError("Incorrect mode: {0}. Must be either 'classification' or 'regression'".format(mode))

    # TODO: in future add here nlp and fundamentals branches
    model, ts_config = ts_model_branch()
    model_config['ts_config'] = ts_config
    model.compile(loss=model_config['loss'], optimizer=model_config['optimizer'], metrics=model_config['metrics'])

    return model, model_config


def ts_model_branch(classification=True):
    c = {'name': 'ts_branch'}
    c['batch_size'] = batch_size
    c['timesteps'] = timesteps
    c['features'] = 5  # TODO: this is a hardcoded value. Need to come up with not hardcoded alternative.
    c['lstm_shapes'] = [100, 50, 20]  # TODO: #2 element is inline with with batch size.
    c['activation'] = 'linear'
    c['go_backwards'] = True
    c['dense_shapes'] = [20, 1]
    c['return_sequence'] = True
    c['stateful'] = True
    c['output'] = 1

    ## Explanations of return_sequence and TimeDistributed: https://stackoverflow.com/questions/42755820/how-to-use-return-sequences-option-and-timedistributed-layer-in-keras
    ## EXplanations of stateful parameter: http://philipperemy.github.io/keras-stateful-lstm/
    ## Good article on different modes of RNN (one-to-many, many-to-many etc.): https://stackoverflow.com/questions/44647258/lstm-autoencoder?noredirect=1&lq=1

    branch = Sequential()
    branch.add(LSTM(c['lstm_shapes'][0], batch_input_shape=(c['batch_size'], c['timesteps'], c['features']),
                    return_sequences=c['return_sequence'], stateful=c['stateful'], name="lstm_1",
                    go_backwards=c['go_backwards']))
    branch.add(LSTM(c['lstm_shapes'][0],
                    return_sequences=c['return_sequence'], name="lstm_2", stateful=c['stateful']))
    branch.add(LSTM(c['lstm_shapes'][0],
                    return_sequences=False, name="lstm_3", stateful=c['stateful']))
    branch.add(Dense(units=c['dense_shapes'][0], activation=c['activation'], name="dense_1"))
    if classification:
        branch.add(Dense(units=c['dense_shapes'][1], activation='sigmoid', name="dense_2"))
    else:
        branch.add(Dense(units=c['dense_shapes'][1], activation=c['activation'], name="dense_2"))

    return branch, c


def fundmentals_model_branch():
    """ branch to handle fundamentals data"""
    branch = Sequential()
    return branch


def nlp_branch_model():
    """branch to handle text and sentiment data"""
    branch = Sequential()
    return branch



def training_model(ticker, epochs, classification, result_folder):
    """method to train the model"""

    # Initialize complex model.
    mode = 'classification' if classification else 'regression'
    model, model_config = get_complex_model(mode=mode)

    model_config['mode'] = mode
    model_config['epochs'] = epochs
    model_config['train_test_val_ratio'] = train_test_val_ratio
    ts_config = model_config['ts_config']

    # Retrieve data and then resize it to fit predefined batch size
    TS_X_train, TS_Y_train, TS_X_test, TS_Y_test, _, _ = \
        get_formated_data(ticker, train_test_val_ratio=train_test_val_ratio, output_shape=
        (ts_config['batch_size'], ts_config['timesteps'], ts_config['features']),
                          classification=classification)

    model.summary()
    print("Inputs: {}".format(model.input_shape))
    print("Outputs: {}".format(model.output_shape))

    # fit network
    # TODO: Not clear, how to sync up batch sizes across multiple branches

    loss = []
    val_loss = []

    for _ in range(model_config['epochs']):
        history = model.fit(TS_X_train, TS_Y_train, epochs=1, batch_size=model_config['ts_config']['batch_size'],
                            validation_data=(TS_X_test, TS_Y_test), verbose=2, shuffle=False)
        loss.append(history.history['loss'])
        val_loss.append(history.history['val_loss'])
        model.reset_states()

    #
    # # plot history
    pyplot.plot(loss, label='train')
    pyplot.plot(val_loss, label='test')
    pyplot.legend()
    pyplot.interactive(False)
    pyplot.savefig(result_folder + "loss_chart.png")
    pyplot.close()

    return model, model_config


def evaluate_model(model, X, Y, result_folder, classification=True, labels=['true', 'predicted']):
    time_series = list(range(X.shape[0]))

    if classification:
        confusion_labels = ['up', 'down']
        Y_pred = model.predict_classes(X, batch_size=batch_size)
        cnf_matrix = confusion_matrix(Y, Y_pred)
        print(cnf_matrix)
        pyplot.imshow(cnf_matrix, interpolation='nearest')
        pyplot.title("Confusion matrix")
        pyplot.colorbar()
        tick_marks = np.arange(2)
        pyplot.xticks(tick_marks, confusion_labels, rotation=45)
        pyplot.yticks(tick_marks, confusion_labels)
        fmt = 'd'
        thresh = cnf_matrix.max() / 2.

        for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
            pyplot.text(j, i, format(cnf_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cnf_matrix[i, j] > thresh else "black")

        pyplot.tight_layout()
        pyplot.ylabel('True label')
        pyplot.xlabel('Predicted label')



    else:
        Y_pred = model.predict(X, batch_size=batch_size)
        pyplot.plot(time_series, Y, label=labels[0])
        pyplot.plot(time_series, Y_pred, label=labels[1])

    pyplot.legend()
    pyplot.interactive(False)
    pyplot.savefig(result_folder + "{0}_vs_{1}.png".format(labels[0], labels[1]))
    pyplot.close()


def get_sample_data(ticker, sample_size):
    file = TIMESERIES_FOLDER + "{0}_df.p".format(ticker)
    with open(file, "rb") as file:
        ticker_df = pickle.load(file)

    random_idx = random.randint(0, (ticker_df.shape[0] - sample_size - 1))

    X_sample = ticker_df.loc[random_idx:(random_idx + sample_size - 1), ['open', 'high', 'low', 'close', 'volume']]
    Y_sample = ticker_df.loc[(random_idx + 1):(random_idx + sample_size), ['close']]

    return X_sample, Y_sample


def _prepare_results_folder(ticker):
    result_folder = './results/{0}_{1}/'.format(ticker, str(uuid.uuid1()))
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    return result_folder


def _dump_config(result_folder, config):
    with open(result_folder + 'model_config.json', 'w') as outfile:
        json.dump(config, outfile)


# TODO: Watch out. Global variables.
batch_size = 64
timesteps = 20
train_test_val_ratio = [0.6, 0.3, 0.1]
classification = False  # if classification True, then model is trained to predict whether
                       # the stock will be up or down. Otherwise, it's trained on regression.
epochs = 20

if __name__ == '__main__':
    # Runtime configs
    ticker = 'GOOG'
    result_folder = _prepare_results_folder(ticker)

    model, config = training_model(ticker, epochs=epochs, classification=classification, result_folder=result_folder)

    TS_X_train, TS_Y_train, TS_X_test, TS_Y_test, TS_X_val, TS_Y_val = \
        get_formated_data(ticker, train_test_val_ratio=[0.6, 0.3, 0.1], output_shape=(batch_size, timesteps, 5),
                          classification=classification)

    evaluate_model(model, X=TS_X_val, Y=TS_Y_val, result_folder=result_folder, labels=['val_true', 'val_predicted'],
                   classification=classification)
    evaluate_model(model, X=TS_X_test, Y=TS_Y_test, result_folder=result_folder, labels=['test_true', 'test_predicted'],
                   classification=classification)

    config['runtime'] = {'ticker': ticker}  # adding some runtime parameters to config dump
    config['model'] = model.to_json()  # adding model description to config dump
    _dump_config(result_folder, config)

    print(result_folder)
