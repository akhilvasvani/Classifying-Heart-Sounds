"""This script runs a LSTM model on Dataset B of Peter Bentley's
Classifying Heart Sounds Challenge."""

from __future__ import print_function

import glob
import os
import csv
import pathlib
import errno
import numpy as np
import librosa

from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical


def extract_data(folder):
    """"
    Using the relevant files from the specified folder path, extract the MFCCs
    (Mel Frequency Cepstral Coefficients) for each file. For each file, save
    40 MFCCs in a data array (as numpy array) and output the data and
    list of filenames.

    Args:
        folder: relative path of where the specific folder of data lies

    Returns:
        data: array of all the MFCCs
        filenames: list of filenames

    Raises:
        OSError: If the file does in fact exist
        FileNotFoundError: If the file is not found
        Exception: Error encountered when parsing the file
        ValueError: If all the MFCCs are 0.
    """

    try:
        filenames = glob.glob(os.path.join(folder, '*.wav'))
    except OSError:
        if pathlib.Path(os.path.join(folder, '*.wav')).resolve(strict=True):
            pass
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    os.path.join(folder, '*.wav'))
    data = []
    mfccs = 0
    for filename in filenames:
        try:
            # Note: Kaiser_fast is faster implementation to load the data
            x, sample_rate = librosa.load(filename, res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate,
                                                 n_mfcc=40).T, axis=0)
        except Exception:
            print("Error encountered while parsing file: ", filename)

        feature = np.array(mfccs).reshape([-1, 1])

        if np.all(feature == 0, axis=0):
            raise ValueError('Error! MFCCs are all 0.')

        data.append(feature)
    return data, filenames


def train_validate_test_split(arr, train_percent=0.75, validate_percent=0.15):
    """
    Splits the array into a training set, validation set, and a test set.

    Args:
        arr: input array to split
        train_percent: how much of the arr should be segmented for the training set.
                       Default is set to 75% of arr
        validate_percent: how much of the arr should be segmented for the
                          validation set. Default is set to 15%.

    Returns:
        train: the training set
        val: the validation set
        test: the testing set
    """

    if train_percent + validate_percent + (1 - (train_percent + validate_percent)) != 1:
        raise ValueError("Error: Training Percent {}, Validation Percent, "
                         "and Testing Percent {} are not 1.".format(train_percent,
                                                                    validate_percent,
                                                                    (1 - (train_percent + validate_percent))))

    train_end = int(train_percent * len(arr))
    validate_end = int(validate_percent * len(arr)) + train_end
    train = arr[:train_end]
    validate = arr[train_end:validate_end]
    test = arr[validate_end:]

    return train, validate, test


def preprocess_data():
    """"
    Preprocesses the data-- extract the MFCCs data for each category
    (normal, murmur, extrasystole), and for each category, label
    via classes (0, 1, 2). Concatenate all the data for each category together
    and split for training, validation, and testing purposes.

    Returns:
        x_train: MFCCs of all 3 categories for training data
        y_train: labels for each respective MFCCs for training data
        x_val: MFCCs of all 3 categories for validation data
        y_val: labels for each respective MFCCs for validation data
        x_test: MFCCs of all 3 categories for testing data
        y_test: labels for each respective MFCCs for testing data
    """

    extrasystole_onehot = 0
    normal_onehot = 1
    murmur_onehot = 2

    normal_sounds, _ = extract_data("normal_train_data")
    normal_labels = [normal_onehot for _ in normal_sounds]

    normal_train, normal_val, normal_test = train_validate_test_split(normal_sounds)
    normal_labels_train, \
    normal_labels_val, \
    normal_labels_test = train_validate_test_split(normal_labels)

    murmur_sounds, _ = extract_data("murmur_train_data")
    murmur_labels = [murmur_onehot for _ in murmur_sounds]

    murmur_train, murmur_val, murmur_test = train_validate_test_split(murmur_sounds)
    murmur_labels_train, \
    murmur_labels_val, \
    murmur_labels_test = train_validate_test_split(murmur_labels)

    extrasystole_sounds, _ = extract_data("extrasystole_train_data")
    extrasystole_labels = [extrasystole_onehot for _ in extrasystole_sounds]

    extrasystole_train, \
    extrasystole_val, extrasystole_test = train_validate_test_split(extrasystole_sounds)
    extrasystole_labels_train, \
    extrasystole_labels_val, \
    extrasystole_labels_test = train_validate_test_split(extrasystole_labels)

    x_train = np.concatenate((normal_train, murmur_train, extrasystole_train))
    y_train = to_categorical(np.concatenate((normal_labels_train,
                                             murmur_labels_train,
                                             extrasystole_labels_train)),
                              num_classes=3)

    x_val = np.concatenate((normal_val, murmur_val, extrasystole_val))
    y_val = to_categorical(np.concatenate((normal_labels_val,
                                           murmur_labels_val,
                                           extrasystole_labels_val)),
                            num_classes=3)

    x_test = np.concatenate((normal_test, murmur_test, extrasystole_test))
    y_test = to_categorical(np.concatenate((normal_labels_test,
                                            murmur_labels_test,
                                            extrasystole_labels_test)),
                             num_classes=3)

    x_predict, x_predict_filenames = extract_data("test_data")

    return x_train, y_train, x_val, y_val, x_test, y_test, np.array(x_predict), x_predict_filenames


def run_model(x_train_data, y_train_data, x_val_data, y_val_data, x_test_data,
              y_test_data, x_predict_data):
    """
    LSTM model that takes in the training data, validation data, and testing
    data. Outputs predictions for unseen data.

    Args:
        x_train_data: MFCC data for all 3 categories for training
        y_train_data: labels for all 3 categories for training
        x_val_data: MFCC data for all 3 categories for validation
        y_val_data: labels for all 3 categories for validation
        x_test_data: MFCC data for all 3 categories for testing
        y_test_data: labels for all 3 categories for testing
        x_predict_data: MFCC for unseen data

    Return:
        predictions: class predictions
    """
    print('Building LSTM Model ...')
    model = Sequential()
    model.add((LSTM(units=64, return_sequences=True,
                    input_shape=(40, 1), activation='tanh')))
    model.add((LSTM(units=32, return_sequences=False,
                    activation='tanh')))
    model.add(Dense(3))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    # sgd = optimizers.SGD(lr=0.002, decay=0.0002/5)
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train_data, y_train_data, batch_size=32, epochs=100,
              validation_data=(x_val_data, y_val_data))
    score = model.evaluate(x_test_data, y_test_data, batch_size=16)
    print("loss: {:3f} - acc: {:3f}".format(score[0], score[1]))
    model.summary()
    model.save("saved_trained_models/LSTM_model.hdf5")

    predictions = model.predict_classes(x_predict_data, batch_size=32)

    return predictions


def write_to_csv(dictionary_results, output_file_path=None):
    """
    Write the dictionary with the filename and respective categorical prediction
    to a csv file.

    Args:
        dictionary_results: dictionary of filenames (keys) and
                            categorical prediction (values).
        output_file_path: path to where to save the file.

    Raises:
        TypeError: If the output_file_path is not specified
    """
    if output_file_path is not None:
        with open(output_file_path+'.csv', mode='w') as output_file:
            writer = csv.writer(output_file, delimiter=',')

            writer.writerow(['Filename', 'Category'])

            for filename, category in dictionary_results.items():
                writer.writerow([filename, category])
    else:
        raise TypeError("Error! output_file_path is not specified.")


def main():
    """
    Main function, which gathers the training, validation, and testing data.
    Attains results from running the LSTM model, and then creates dictionary
    with the filename as a key, and the predicted category as a value.
    Finally, writes to a csv file.

    Raises:
        OSError: If the file does in fact exist
        FileNotFoundError: If the file is not found
    """
    x_train, y_train, x_val, y_val, x_test, y_test, x_predict, x_predict_filenames = preprocess_data()
    results = run_model(x_train, y_train, x_val, y_val, x_test, y_test, x_predict)

    true_results_dict = {}

    for filename, number in list(zip(x_predict_filenames, results)):
        _, tail = os.path.split(filename)
        if number == 1:
            true_results_dict[tail] = 'normal'
        elif number == 2:
            true_results_dict[tail] = 'murmur'
        else:
            true_results_dict[tail] = 'extrasystole'

    try:
        filepath = os.path.join('results', 'LSTM_predictions')
    except OSError:
        if pathlib.Path(os.path.join('results', 'LSTM_predictions')).resolve(strict=True):
            pass
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    os.path.join('results', 'LSTM_predictions'))

    write_to_csv(true_results_dict, output_file_path=filepath)


if __name__ == '__main__':
    main()
