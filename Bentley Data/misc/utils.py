import numpy as np
import matplotlib.pyplot as plt
from threading import Thread


def get_S1S2_bounds(data, peaks, set_name):

    # Finds the differences between all peaks in every file
    all_diffs = [np.diff(peaks[i]) for i in range(len(peaks))]

    # finding max difference or diastole period
    # and then labelling the first peak as s2 and second peak as s1
    max_index = []
    s1s2_peaks = []
    for j, item in enumerate(all_diffs):
        if any(all_diffs[j]):
            max_index.append(np.argmax(all_diffs[j]))
            s2 = peaks[j][max_index[j]]
            s1 = peaks[j][max_index[j] + 1]
            s1s2_peaks.append([s1, s2])
        else:
            max_index.append(-1)
            s1s2_peaks.append([-1, -1])
    s1s2_peaks = np.array(s1s2_peaks)

    # defining s1 and s2 boundaries
    s1_bounds = []
    s2_bounds = []
    if set_name == 'A':
        upper_s1, lower_s1, upper_s2, lower_s2 = np.array([200, 80, 600, 70]) * 2
    else:
        upper_s1, lower_s1, upper_s2, lower_s2 = np.array([25, 10, 35, 10]) * 10

    for k, item in enumerate(s1s2_peaks):
        if s1s2_peaks[k][0] == -1:
            s1_bounds.append([-1, -1])
            s2_bounds.append([-1, -1])
        else:
            s1_lower, s1_upper = s1s2_peaks[k][0] - lower_s1, s1s2_peaks[k][0] + upper_s1
            s2_lower, s2_upper = s1s2_peaks[k][1] - lower_s2, s1s2_peaks[k][1] + upper_s2
            if s1_lower < 0:
                s1_lower = 0
            if s2_lower < 0:
                s2_lower = 0
            if s1_upper >= len(data[0]):
                s1_upper = len(data[0]) - 1
            if s2_upper >= len(data[0]):
                s2_upper = len(data[0]) - 1
            s1_bounds.append([s1_lower, s1_upper])
            s2_bounds.append([s2_lower, s2_upper])

    return np.array(s1_bounds), np.array(s2_bounds)


def featurePlot(feat, set_name, label, **kwargs):
    """function for plotting class specific points of a feature"""
    if set_name.upper() == 'A':
        print("Red=artifacts, blue=extrahls, purple=murmur, cyan=normal")
        y_labelA = label
        for i, item in enumerate(feat):
            if y_labelA[i] == 'artifact':
                plt.scatter(i, feat[i], c='r')
            elif y_labelA[i] == 'extrahls':
                plt.scatter(i, feat[i], c='b')
            elif y_labelA[i] == 'murmur':
                plt.scatter(i, feat[i], c='m')
            elif y_labelA[i] == 'normal':
                plt.scatter(i, feat[i], c='c')
        plt.title(str(kwargs.values()))
        plt.savefig(str(kwargs.values())+'.png')
    else:
        print("blue=extrastole, purple=murmur, cyan=normal")
        y_labelB = label
        for i, item in enumerate(feat):
            if y_labelB[i] == 'extrastole':
                plt.scatter(i, feat[i], c='k')
            elif y_labelB[i] == 'murmur':
                plt.scatter(i, feat[i], c='m')
            elif y_labelB[i] == 'normal':
                plt.scatter(i, feat[i], c='c')
        plt.title(kwargs.values())
        plt.show()
    return


def stdInterval(lower, low_index, upper, up_index, data):
    """ std deviation of specific interval where lower is the left most bound of the interval,
        upper is right most bound. """
    std = []
    for k, item in enumerate(data):
        if lower[k][0] == -1:
            std.append(0)
        else:
            dev = np.std(data[k][lower[k][low_index]:upper[k][up_index]])
            if np.isnan(dev):
                std.append(0)
            else:
                std.append(dev)
    return np.array(std)


def normalize(a):
    if isinstance(a, np.ndarray):
        return a / max(a)
    else:
        raise TypeError('Error: not a numpy array')


def freqInterval(data, lower, l_index, upper, u_index):
    freq = []
    for i, item in enumerate(data):
        if lower[i][0] == -1:
            freq.append(0)
        else:
            temp = data[i][lower[i][l_index]:upper[i][u_index]]
            temp = np.fft.fft(temp)
            temp = np.abs(temp)/max(np.abs(temp))
            #freq.append(temp[:int(len(temp)/2)])
            freq.append(temp)
    return np.array(freq)


class ThreadWithReturnValue(Thread):
    """
    Created a Thread subclass. It is a workable doaround,
    but it accesses "private" data structures that are specific to Thread implementation, so
    things will get a little hairy.
    """

    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, Verbose=None):
        """ Initializes the thread object. """

        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        """ Runs the function if specified for the thread. """
        # If the target function is specified
        if self._target is not None:
            # Run the function
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        """ Returns the value of target function running in the thread. """

        Thread.join(self, *args)
        return self._return
