# imports
import numpy as np
import matplotlib.pyplot as plt
import audioFeatureExtraction
import csv
import sys

from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from python_speech_features import mfcc

from preprocessing import get_preprocessed_data, get_test_data
from features import find_peaks

sys.path.append('misc')
from utils import get_S1S2_bounds, stdInterval, freqInterval, featurePlot, ThreadWithReturnValue, normalize


def set_A(x_dataA, x_testdataA, y_labelA, if_plot=False):
    """Get peaks for A"""
    dataA_peaks = find_peaks(x_dataA, 'A')
    testdataA_peaks = find_peaks(x_testdataA, 'A')

    # Get the S1 and S2 bounds for set A
    s1_boundsA, s2_boundsA = get_S1S2_bounds(x_dataA, dataA_peaks, 'A')
    s1_boundstA, s2_boundstA = get_S1S2_bounds(x_testdataA, testdataA_peaks, 'A')

    # Standard deviation of S1
    stdS1_A = normalize(stdInterval(s1_boundsA, 0, s1_boundsA, 1, x_dataA))
    stdS1_testA = normalize(stdInterval(s1_boundstA, 0, s1_boundstA, 1, x_testdataA))

    # Standard deviation of S2
    stdS2_A = normalize(stdInterval(s2_boundsA, 0, s2_boundsA, 1, x_dataA))
    stdS2_testA = normalize(stdInterval(s2_boundstA, 0, s2_boundstA, 1, x_testdataA))

    # frequency intervals of S1 and S2
    freqS1_A = freqInterval(x_dataA, s1_boundsA, 0, s1_boundsA, 1)
    freqS2_A = freqInterval(x_dataA, s2_boundsA, 0, s2_boundsA, 1)

    # standard deviation of S1 and S2 frequencies
    stdS1_freqA = normalize(np.array([np.std(f) for f in freqS1_A]))
    stdS2_freqA = normalize(np.array([np.std(f) for f in freqS2_A]))

    # mean of S1 and S2 frequencies
    meanS1_freqA = normalize(np.array([np.average(f) for f in freqS1_A]))
    meanS2_freqA = normalize(np.array([np.average(f) for f in freqS2_A]))

    if if_plot:
        featurePlot(stdS1_A, 'A', y_labelA, title='Standard Deviation of S1 (set A) vs. Feature Labels (set A)')
        featurePlot(stdS2_A, 'A', y_labelA, title='Standard Deviation of S2 (set A) vs. Feature Labels (set A)')

        featurePlot(stdS1_freqA, 'A', y_labelA, title='Standard Deviation of Frequency of S1 (set A) vs. Feature Labels'
                                                      ' (set A)')
        featurePlot(meanS1_freqA, 'A', y_labelA, title='Mean of Frequency of S1 (set A) vs. Feature Labels (set A)')
        featurePlot(stdS2_freqA, 'A', y_labelA, title='Standard Deviation of Frequency of S2 (set A) vs. Feature Labels'
                                                      ' (set A)')
        featurePlot(meanS2_freqA, 'A', y_labelA, title='Mean of Frequency of S2 (set A) vs. Feature Labels (set A)')

        n = 2
        plt.figure()
        plt.title(y_labelA[n])
        plt.plot(x_dataA[n], 'b')
        plt.scatter(dataA_peaks[n], x_dataA[n][dataA_peaks[n]], c='r')
        # plt.show()
        plt.savefig(y_labelA[n]+'.png')

    return stdS1_A, stdS1_testA, stdS2_A, stdS2_testA, meanS1_freqA, meanS2_freqA, stdS1_freqA, stdS2_freqA


def set_B(x_dataB, y_labelB, if_plot=False):
    """get peaks for B"""
    dataB_peaks = find_peaks(x_dataB, 'B')

    # Get S1 and S2 bounds for set B
    s1_boundsB, s2_boundsB = get_S1S2_bounds(x_dataB, dataB_peaks, 'B')

    # Standard deviation of S1
    stdS1_B = normalize(stdInterval(s1_boundsB, 0, s1_boundsB, 1, x_dataB))

    # Standard deviation of S2
    stdS2_B = normalize(stdInterval(s2_boundsB, 0, s2_boundsB, 1, x_dataB))

    # frequency intervals of S1 and S2
    freqS1_B = freqInterval(x_dataB, s1_boundsB, 0, s1_boundsB, 1)
    freqS2_B = freqInterval(x_dataB, s2_boundsB, 0, s2_boundsB, 1)

    # standard deviation of S1 and S2 frequencies
    stdS1_freqB = normalize(np.array([np.std(f) for f in freqS1_B]))
    stdS2_freqB = normalize(np.array([np.std(f) for f in freqS2_B]))

    # mean of S1 and S2 frequencies
    meanS1_freqB = normalize(np.array([np.average(f) for f in freqS1_B]))
    meanS2_freqB = normalize(np.array([np.average(f) for f in freqS2_B]))

    if if_plot:
        featurePlot(stdS1_B, 'B', y_labelB, title='Standard Deviation of S1 (set B) vs. Feature Labels (set B)')
        featurePlot(stdS2_B, 'B', y_labelB, title='Standard Deviation of S2 (set B) vs. Feature Labels (set B)')

        featurePlot(stdS1_freqB, 'B', y_labelB, title='Standard Deviation of Frequency of S1 (set B) vs. Feature Labels'
                                                      ' (set B)')
        featurePlot(stdS2_freqB, 'B', y_labelB, title='Standard Deviation of Frequency of S2 (set B) vs. Feature Labels'
                                                      ' (set B)')
        featurePlot(meanS1_freqB, 'B', y_labelB, title='Mean of Frequency of S1 (set B) vs. Feature Labels (set B)')
        featurePlot(meanS2_freqB, 'B', y_labelB, title='Mean of Frequency of S2 (set B) vs. Feature Labels (set B)')

        n = 390
        plt.figure()
        plt.title(y_labelB[n])
        plt.plot(x_dataB[n], 'b')
        plt.scatter(dataB_peaks[n], x_dataB[n][dataB_peaks[n]], c='r')
        plt.show()

    return stdS1_B, stdS2_B, meanS1_freqB, meanS2_freqB, stdS1_freqB, stdS2_freqB


def zero_crossing(x_dataA, x_dataB, x_testdataA, y_labelA, y_labelB, if_plot=False):
    """zero crossing rate of frame"""
    zero_crossingsA = normalize(np.array([audioFeatureExtraction.stZCR(x) for x in x_dataA]))
    zero_crossingsB = normalize(np.array([audioFeatureExtraction.stZCR(x) for x in x_dataB]))
    zc_testA = normalize(np.array([audioFeatureExtraction.stZCR(x) for x in x_testdataA]))

    if if_plot:
        featurePlot(zero_crossingsA, 'A', y_labelA, title='Zero Crossings Rate of Frame (set A) vs Feature Labels'
                                                          ' (set A)')
        featurePlot(zero_crossingsB, 'B', y_labelB, title='Zero Crossings Rate of Frame (set B) vs Feature Labels'
                                                          ' (set B)')

    return zero_crossingsA, zero_crossingsB, zc_testA


def signal_energy_frame(x_dataA, x_dataB, x_testdataA, y_labelA, y_labelB, if_plot=False):
    """signal energy of frame"""
    energyA = normalize(np.array([audioFeatureExtraction.stEnergy(x) for x in x_dataA]))
    energyB = normalize(np.array([audioFeatureExtraction.stEnergy(x) for x in x_dataB]))
    ener_testA = normalize(np.array([audioFeatureExtraction.stEnergy(x) for x in x_testdataA]))

    if if_plot:
        featurePlot(energyA, 'A', y_labelA, title='Signal Energy of Frame (set A) vs Feature Labels (set A)')
        featurePlot(energyB, 'B', y_labelB, title='Signal Energy of Frame (set B) vs Feature Labels (set B)')

    return energyA, energyB, ener_testA


def entropy_of_energy(x_dataA, x_dataB, x_testdataA, y_labelA, y_labelB, if_plot=False):
    """Entropy of Energy"""
    entropyA = normalize(np.array([audioFeatureExtraction.stEnergyEntropy(x, numOfShortBlocks=50) for x in x_dataA]))
    entropyB = normalize(np.array([audioFeatureExtraction.stEnergyEntropy(x, numOfShortBlocks=50) for x in x_dataB]))
    entr_testA = normalize(np.array([audioFeatureExtraction.stEnergyEntropy(x, numOfShortBlocks=50)
                                     for x in x_testdataA]))

    if if_plot:
        featurePlot(entropyA, 'A', y_labelA, title='Entropy of Energy (set A) vs Feature Labels (set A)')
        featurePlot(entropyB, 'B', y_labelB, title='Entropy of Energy (set B) vs Feature Labels (set B)')

    return entropyA, entropyB, entr_testA


def frequency_domain(x_dataA, x_dataB, x_testdataA, if_plot=False):
    """Get the frequency domain of signal"""

    X_dataA = np.array([np.fft.fft(x) for x in x_dataA])
    X_dataA = np.array([np.abs(X) / max(np.abs(X)) for X in X_dataA])

    X_testdataA = np.array([np.fft.fft(x) for x in x_testdataA])
    X_testdataA = np.array([X[:len(X_dataA[0])] for X in X_testdataA])
    X_testdataA = np.array([np.abs(X) / max(np.abs(X)) for X in X_testdataA])
    X_testdataA = np.array([X[:len(X) - 1] if len(X) % 2 == 1 else X for X in X_testdataA])

    # Xn_dataA = np.array([nfft(x,2) for x in x_dataA])
    # Xn_dataA = np.array([np.abs(X)/max(np.abs(X)) for X in Xn_dataA])

    X_dataB = np.array([np.fft.fft(x) for x in x_dataB])
    X_dataB = np.array([np.abs(X) / max(np.abs(X)) for X in X_dataB])

    if if_plot:
        plt.figure()
        plt.title('Normalized Amplitude (Set A) vs Frequency')
        plt.plot(X_dataA[70])
        plt.show()

        plt.figure()
        plt.title('Normalized Amplitude (Set B) vs Frequency')
        plt.plot(X_dataB[300])
        plt.show()

        # plt.figure()
        # plt.plot(Xn_dataA[70])
        # plt.show()

    return X_dataA, X_dataB, X_testdataA


def spectral_entropy(X_dataA, X_dataB, X_testdataA, y_labelA, y_labelB, if_plot=False):
    """Get the Spectral entropy"""

    entropy_freqA = normalize(np.array([audioFeatureExtraction.stSpectralEntropy(X, numOfShortBlocks=150)
                                        for X in X_dataA]))
    entropy_freqB = normalize(np.array([audioFeatureExtraction.stSpectralEntropy(X, numOfShortBlocks=50)
                                        for X in X_dataB]))
    entr_freqtestA = normalize(np.array([audioFeatureExtraction.stSpectralEntropy(X, numOfShortBlocks=150)
                                         for X in X_testdataA]))

    if if_plot:
        featurePlot(entropy_freqA, 'A', y_labelA, title='Spectral Entropy (set A) vs Feature Labels (set A)')
        featurePlot(entropy_freqB, 'B', y_labelB, title='Spectral Entropy (set B) vs Feature Labels (set B)')

    return entropy_freqA, entropy_freqB, entr_freqtestA


def spectral_flux(X_dataA, X_dataB, X_testdataA, y_labelA, y_labelB, if_plot=False):
    """ Get spectral flux """
    fluxA = normalize(np.array([np.abs(audioFeatureExtraction.stSpectralFlux(X[:int(len(X) / 2)], X[int(len(X) / 2):]))
                                for X in X_dataA]))
    fluxB = normalize(np.array([np.abs(audioFeatureExtraction.stSpectralFlux(X[:int(len(X) / 2)],
                                                                             X[int(len(X) / 2) + 1:])) for X in X_dataB]))
    flux_testA = normalize(np.array([np.abs(audioFeatureExtraction.stSpectralFlux(X[:int(len(X) / 2)],
                                                                                  X[int(len(X) / 2):])) for X in X_testdataA]))

    if if_plot:
        featurePlot(fluxA, 'A', y_labelA, title='Spectral Flux (set A) vs Feature Labels (set A)')
        featurePlot(fluxB, 'B', y_labelB, title='Spectral Flux (set B) vs Feature Labels (set B)')

    return fluxA, fluxB, flux_testA


def spectral_centroid_frame(X_dataA, X_dataB, X_testdataA, y_labelA, y_labelB, framerate_A, framerate_B, if_plot=False):
    """Get Spectral centroid of frame (given abs(FFT))"""

    FsA = int(framerate_A[0] / 20)  # framerate is the same for all of set A
    FsB = int(framerate_B[0] / 10)  # framerate is the same for all of set B

    centroidA = np.array([audioFeatureExtraction.stSpectralCentroidAndSpread(np.abs(X), FsA) for X in X_dataA])
    centroidA[:, 0] = normalize(centroidA[:, 0])
    centroidA[:, 1] = normalize(centroidA[:, 1])

    centroidB = np.array([audioFeatureExtraction.stSpectralCentroidAndSpread(np.abs(X), FsB) for X in X_dataB])
    centroidB[:, 0] = normalize(centroidB[:, 0])
    centroidB[:, 1] = normalize(centroidB[:, 1])

    cent_testA = np.array([audioFeatureExtraction.stSpectralCentroidAndSpread(np.abs(X), FsA) for X in X_testdataA])
    cent_testA[:, 0] = normalize(cent_testA[:, 0])
    cent_testA[:, 1] = normalize(cent_testA[:, 1])

    if if_plot:
        featurePlot(centroidA[:, 0], 'A', y_labelA, title='Centroids (1st Column of Set A) vs. Feature Labels (Set A)')
        featurePlot(centroidA[:, 1], 'A', y_labelA, title='Centroids (2nd Column of Set A) vs. Feature Labels (Set A)')
        featurePlot(centroidB[:, 0], 'B', y_labelB, title='Centroids (1st Column of Set B) vs. Feature Labels (Set B)')
        featurePlot(centroidB[:, 1], 'B', y_labelB, title='Centroids (2nd Column of Set B) vs. Feature Labels (Set B)')

    return centroidA, centroidB, cent_testA


def get_mcfcc_feat(x_dataA, x_dataB, x_testdataA, framerate_A, framerate_B, if_plot=False):
    """Get the MFCC features"""

    FsA = int(framerate_A[0] / 10)  # framerate is the same for all of set A
    FsB = int(framerate_B[0] / 4)  # framerate is the same for all of set B

    mfccA = np.array([mfcc(np.abs(x), samplerate=FsA, numcep=3, winlen=0.025) for x in x_dataA])
    mfccA_feat = np.array([[np.average(m[:, 0]), np.std(m[:, 0]), np.average(m[:, 1]), np.std(m[:, 1]),
                            np.average(m[:, 2]), np.std(m[:, 2])] for m in mfccA])
    for i in range(len(mfccA_feat[0])):
        mfccA_feat[:, i] = np.abs(mfccA_feat[:, i]) / max(np.abs(mfccA_feat[:, i]))

    mfccB = np.array([mfcc(np.abs(x), samplerate=FsB, numcep=5, winlen=0.01, winstep=0.01) for x in x_dataB])
    mfccB_feat = np.array([[np.average(m[:, 0]), np.std(m[:, 0]), np.average(m[:, 1]), np.std(m[:, 1]),
                            np.average(m[:, 2]), np.std(m[:, 2])] for m in mfccB])
    for i in range(len(mfccB_feat[0])):
        mfccB_feat[:, i] = np.abs(mfccB_feat[:, i]) / max(np.abs(mfccB_feat[:, i]))

    mfcc_testA = np.array([mfcc(np.abs(x), samplerate=FsA, numcep=3, winlen=0.025) for x in x_testdataA])
    mfcctestA_feat = np.array([[np.average(m[:, 0]), np.std(m[:, 0]), np.average(m[:, 1]), np.std(m[:, 1]),
                                 np.average(m[:, 2]), np.std(m[:, 2])] for m in mfcc_testA])
    for i in range(len(mfccA_feat[0])):
        mfccA_feat[:, i] = np.abs(mfccA_feat[:, i]) / max(np.abs(mfccA_feat[:, i]))

    if if_plot:
        featurePlot(mfccA_feat[:, 0], 'A', title='MFCC Features (1st column) (Set A) vs Features Labels (Set A)')
        featurePlot(mfccA_feat[:, 1], 'A', title='MFCC Features (2nd column) (Set A) vs Features Labels (Set A)')
        featurePlot(mfccB_feat[:, 0], 'B', title='MFCC Features (1st column) (Set B) vs Features Labels (Set B)')
        featurePlot(mfccB_feat[:, 1], 'B', title='MFCC Features (2nd column) (Set B) vs Features Labels (Set B)')

    return mfccA_feat, mfccB_feat, mfcctestA_feat


def run_model_A(method, kf, x_trainA, y_trainA, priorsA):
    """ Run several ML algorithms for classifying S1 and S2 into the 4 categories """

    resultsA = []
    modelA = []  # the models are saved here
    i = 0
    precisionsA = {k: [] for k in range(4)}
    p_modelA = []
    r_modelA = []

    for train_index, test_index in kf.split(x_trainA):
        cm = np.zeros((4, 4))
        TP_A = {k: 0 for k in range(4)}
        FP_A = {k: 0 for k in range(4)}
        if method in ['GaussianNB', 'AdaBoostClassifier']:
            modelA.append(GaussianNB(priorsA))
        elif method == 'SVM':
            modelA.append(SVC(class_weight={1: 8, 2: 3, 3: 9}, gamma=1, C=1))
        elif method == 'DecisionTreeClassifer':
            modelA.append(DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=3))
        elif method == 'RandomForestClassifier':
            modelA.append(RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=3, min_samples_split=3))
        else:  # method == 'GradientBoostingClassifier'
            modelA.append(GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3))

        resultsA.append(modelA[i].fit(x_trainA[train_index], y_trainA[train_index]).score(x_trainA[test_index],
                                                                                          y_trainA[test_index]))
        preds = modelA[i].predict(x_trainA[test_index])
        actual = y_trainA[test_index]

        for p, item in enumerate(preds):
            cm[actual[p]][preds[p]] += 1
            if preds[p] == actual[p]:
                TP_A[preds[p]] += 1
            else:
                FP_A[preds[p]] += 1
        for n in range(4):
            if TP_A[n] == 0 and FP_A[n] == 0:
                precisionsA[n].append(0)
            else:
                precisionsA[n].append(float(TP_A[n] / (TP_A[n] + FP_A[n])))

        TP = cm[0][0] + cm[1][1] + cm[2][2] + cm[3][3]
        FP = np.sum(cm[1:, 0]) + np.sum(cm[0:1, 1]) + np.sum(cm[2:, 1]) + np.sum(cm[0:2, 2] + cm[3:, 2]) + \
             np.sum(cm[0:3, 3])
        FN = np.sum(cm[0, 1:]) + np.sum(cm[1, 0:1]) + np.sum(cm[1, 2:]) + np.sum(cm[2, 0:2]) + np.sum(cm[2, 3:]) + \
             np.sum(cm[3, 0:3])
        p_modelA.append(float(TP / (TP + FP)))
        r_modelA.append(float(TP / (TP + FN)))
        i += 1
    resultsA = np.array(resultsA)

    # write this out to a CSV file
    with open('Results/'+method+'_A.csv', mode='w') as file_a:
        writer = csv.writer(file_a, delimiter=',')
        writer.writerow(['Overall Average Precision: ', np.average(p_modelA), ' breakdown: ', p_modelA])
        writer.writerow(['Overall Average Recall: ', np.average(r_modelA), ' breakdown: ', r_modelA])
        writer.writerow(['Precisions: ', precisionsA])
        writer.writerow(['Average Precisions for each class: ', np.average(precisionsA[0]), np.average(precisionsA[1]),
                        np.average(precisionsA[2]), np.average(precisionsA[3])])
        writer.writerow(['Accuracy of splits: ', resultsA])
        writer.writerow(['Average results: ', np.average(resultsA)])
    file_a.close()


def run_model_B(method, kf, x_trainB, y_trainB,  priorsB):
    """ Run several ML algorithms for classifying S1 and S2 into the 4 categories """
    resultsB = []
    modelB = []  # the models are saved here
    i = 0
    precisionsB = {k: [] for k in range(3)}
    p_modelB = []
    r_modelB = []

    for train_index, test_index in kf.split(x_trainB):
        cm = np.zeros((3, 3))
        TP_B = {k: 0 for k in range(3)}
        FP_B = {k: 0 for k in range(3)}
        if method == 'GaussianNB':
            modelB.append(GaussianNB(priorsB))
        elif method == 'AdaBoostClassifier':
            modelB.append(AdaBoostClassifier(base_estimator=GaussianNB(), learning_rate=0.001, n_estimators=100))
        elif method == 'SVM':
            modelB.append(SVC(class_weight={0: 20, 1: 5, 2: 10}, C=1, gamma=1))
        elif method == 'DecisionTreeClassifer':
            modelB.append(DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=3))
        elif method == 'RandomForestClassifer':
            modelB.append(RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=1,
                                                 min_samples_split=3))
        else:  # method == GradientBoostingClassifer
            modelB.append(GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=15))

        resultsB.append(modelB[i].fit(x_trainB[train_index], y_trainB[train_index]).score(x_trainB[test_index],
                                                                                          y_trainB[test_index]))
        preds = modelB[i].predict(x_trainB[test_index])
        actual = y_trainB[test_index]

        for p, item in enumerate(preds):
            cm[actual[p]][preds[p]] += 1
            if preds[p] == actual[p]:
                TP_B[preds[p]] += 1
            else:
                FP_B[preds[p]] += 1
        for n in range(3):
            if TP_B[n] == 0 and FP_B[n] == 0:
                precisionsB[n].append(0)
            else:
                precisionsB[n].append(float(TP_B[n] / (TP_B[n] + FP_B[n])))
        TP = cm[0][0] + cm[1][1] + cm[2][2]
        FP = np.sum(cm[1:, 0]) + np.sum(cm[0:1, 1]) + np.sum(cm[2:, 1]) + np.sum(cm[0:2, 2])
        FN = np.sum(cm[0, 1:]) + np.sum(cm[1, 0:1]) + np.sum(cm[1, 2:]) + np.sum(cm[2, 0:2])
        p_modelB.append(float(TP / (TP + FP)))
        r_modelB.append(float(TP / (TP + FN)))
        i += 1
    resultsB = np.array(resultsB)

    with open('Results/'+method+'_B.csv', mode='w') as file_b:
        writer = csv.writer(file_b, delimiter=',')
        writer.writerow(['Overall Average Precision: ', np.average(p_modelB), ' breakdown: ', p_modelB])
        writer.writerow(['Overall Average Recall: ', np.average(r_modelB), ' breakdown: ', r_modelB])
        writer.writerow(['Precisions: ', precisionsB])
        writer.writerow(['Average Precisions for each class: ', np.average(precisionsB[0]), np.average(precisionsB[1]),
                         np.average(precisionsB[2])])
        writer.writerow(['Accuracy of splits: ', resultsB])
        writer.writerow(['Average results: ', np.average(resultsB)])
    file_b.close()


def main():
    """ The main script. Runs all of the above functions. and saves in the result folder all of the accuracies
        and predictions for each ML algorithm.
        To plot any the following function --> set if_plot to True in each of the function arguments."""

    # Get the preprocesed data
    x_dataA, y_labelA, framerate_A = get_preprocessed_data('A', N=2, factor=5)
    x_dataB, y_labelB, framerate_B = get_preprocessed_data('B', N=2, factor=1)

    # Get the testing data
    x_testdataA = get_test_data('A')
    x_testdataA = np.array([x[:len(x_dataA[0])] for x in x_testdataA])

    stdS1_A, stdS1_testA, stdS2_A, stdS2_testA, meanS1_freqA, meanS2_freqA, stdS1_freqA, stdS2_freqA = set_A(x_dataA,
                                                                                                             x_testdataA,
                                                                                                             y_labelA)

    stdS1_B, stdS2_B, meanS1_freqB, meanS2_freqB, stdS1_freqB, stdS2_freqB = set_B(x_dataB, y_labelB)

    # Opted to use Multi-thread to speed process up
    twrv1 = ThreadWithReturnValue(target=zero_crossing, args=(x_dataA, x_dataB, x_testdataA, y_labelA, y_labelB))
    twrv2 = ThreadWithReturnValue(target=signal_energy_frame, args=(x_dataA, x_dataB, x_testdataA, y_labelA, y_labelB))
    twrv3 = ThreadWithReturnValue(target=entropy_of_energy, args=(x_dataA, x_dataB, x_testdataA, y_labelA, y_labelB))
    twrv4 = ThreadWithReturnValue(target=frequency_domain, args=(x_dataA, x_dataB, x_testdataA))

    twrv1.start()
    twrv2.start()
    twrv3.start()
    twrv4.start()

    [zero_crossingsA, zero_crossingsB, zc_testA] = twrv1.join()
    [energyA, energyB, ener_testA] = twrv2.join()
    [entropyA, entropyB, entr_testA] = twrv3.join()
    [X_dataA, X_dataB, X_testdataA] = twrv4.join()

    twrv5 = ThreadWithReturnValue(target=spectral_entropy, args=(X_dataA, X_dataB, X_testdataA, y_labelA, y_labelB))
    twrv6 = ThreadWithReturnValue(target=spectral_flux, args=(X_dataA, X_dataB, X_testdataA, y_labelA, y_labelB))
    twrv7 = ThreadWithReturnValue(target=spectral_centroid_frame, args=(X_dataA, X_dataB, X_testdataA, y_labelA,
                                                                        y_labelB, framerate_A, framerate_B))
    twrv8 = ThreadWithReturnValue(target=get_mcfcc_feat, args=(x_dataA, x_dataB, x_testdataA, framerate_A, framerate_B))

    twrv5.start()
    twrv6.start()
    twrv7.start()
    twrv8.start()

    [entropy_freqA, entropy_freqB, entr_freqtestA] = twrv5.join()
    [fluxA, fluxB, flux_testA] = twrv6.join()
    [centroidA, centroidB, cent_testA] = twrv7.join()
    [mfccA_feat, mfccB_feat, mfcctestA_feat] = twrv8.join()

    # features to use
    # zero_crossings, energy, entropy, entropy frequency, flux, spread, mfcc #fluxA, flux_testA, fluxB
    x_utrainA = np.column_stack((zero_crossingsA, energyA, entropyA, entropy_freqA, fluxA, centroidA[:, 1], mfccA_feat,
                                 stdS1_A, stdS2_A, stdS1_freqA, meanS1_freqA, stdS2_freqA, meanS2_freqA))
    x_utestA = np.column_stack((zc_testA, ener_testA, entr_testA, entr_freqtestA, flux_testA, cent_testA[:, 1],
                                mfcctestA_feat, stdS1_testA, stdS2_testA))

    x_utrainB = np.column_stack((zero_crossingsB, energyB, entropyB, entropy_freqB, fluxB, centroidB[:, 1], mfccB_feat,
                                 stdS1_B, stdS2_B, stdS1_freqB, meanS1_freqB, stdS2_freqB, meanS2_freqB))

    le = preprocessing.LabelEncoder()
    y_utrainA = le.fit_transform(y_labelA)  # 0 - artifact, 1 - extrahls, 2 - murmur, 3 - normal
    y_utrainB = le.fit_transform(y_labelB)

    # shuffle data
    x_trainA, y_trainA = shuffle(x_utrainA, y_utrainA, random_state=3)
    x_trainB, y_trainB = shuffle(x_utrainB, y_utrainB, random_state=2)

    # split kfold data
    kf = KFold(n_splits=4, shuffle=False, random_state=0)

    # priors of each class
    priorsA = [i / len(x_trainA) for i in np.bincount(y_trainA)]
    priorsB = [i / len(x_trainB) for i in np.bincount(y_trainB)]

    twrv9 = ThreadWithReturnValue(target=run_model_A, args=('GaussianNB', kf, x_trainA, y_trainA, priorsA))
    twrv10 = ThreadWithReturnValue(target=run_model_A, args=('AdaBoostClassifier', kf, x_trainA, y_trainA, priorsA))
    twrv11 = ThreadWithReturnValue(target=run_model_A, args=('SVM', kf, x_trainA, y_trainA, priorsA))
    twrv12 = ThreadWithReturnValue(target=run_model_A, args=('DecisionTreeClassifier', kf, x_trainA, y_trainA, priorsA))
    twrv13 = ThreadWithReturnValue(target=run_model_A, args=('RandomForestClassifier', kf, x_trainA, y_trainA, priorsA))
    twrv14 = ThreadWithReturnValue(target=run_model_A, args=('GradientBoostingClassifier', kf, x_trainA, y_trainA, priorsA))

    twrv15 = ThreadWithReturnValue(target=run_model_B, args=('GaussianNB', kf, x_trainB, y_trainB, priorsB))
    twrv16 = ThreadWithReturnValue(target=run_model_B, args=('AdaBoostClassifier', kf, x_trainB, y_trainB, priorsB))
    twrv17 = ThreadWithReturnValue(target=run_model_B, args=('SVM', kf, x_trainB, y_trainB, priorsB))
    twrv18 = ThreadWithReturnValue(target=run_model_B, args=('DecisionTreeClassifier', kf, x_trainB, y_trainB, priorsB))
    twrv19 = ThreadWithReturnValue(target=run_model_B, args=('RandomForestClassifier', kf, x_trainB, y_trainB, priorsB))
    twrv20 = ThreadWithReturnValue(target=run_model_B, args=('GradientBoostingClassifier', kf, x_trainB, y_trainB, priorsB))

    twrv9.start()
    twrv10.start()
    twrv11.start()
    twrv12.start()
    twrv13.start()
    twrv14.start()
    twrv15.start()
    twrv16.start()
    twrv17.start()
    twrv18.start()
    twrv19.start()
    twrv20.start()

    twrv9.join()
    twrv10.join()
    twrv11.join()
    twrv12.join()
    twrv13.join()
    twrv14.join()
    twrv15.join()
    twrv16.join()
    twrv17.join()
    twrv18.join()
    twrv19.join()
    twrv20.join()


if __name__ == '__main__':
    main()
