import csv
import numpy as np
import wave
import struct
import pywt


def moreNsecs(name_of_file, N):
	""" Ignores all entries that are less than N seconds """
	
	f = wave.open(name_of_file)
	frames = f.readframes(-1)
	samples = struct.unpack('h'*f.getnframes(), frames)
	framerate = f.getframerate()
	t = [float(i)/framerate for i in range(len(samples))]
	return t[-1] > 2


def get_framerate(name_of_file):
	""" Returns framerate """
	f = wave.open(name_of_file)
	return f.getframerate()


def get_setA_file_label(N):
	""" Returns set a file and label.
		Input: N - minimum file length
	"""

	x_trainAFile = []  # wave filename
	y_trainA = []  # label
	framerate = []  # frame rate
	with open('Data/set_a.csv') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			if not row[2] == '' and not row[2] == 'label':
				if moreNsecs(row[1], 2):
					x_trainAFile.append(row[1])
					y_trainA.append(row[2])
					framerate.append(get_framerate(row[1]))
	return np.array(x_trainAFile), np.array(y_trainA), np.array(framerate)


def get_setA_testfile(N):
	with open('Data/set_a.csv') as csvfile:
		reader = csv.reader(csvfile)
		x_testFile = [row[1][:11] + 'Aunlabelledtest' + row[1][11:] for row in reader if row[2] == '']
	return np.array(x_testFile)


def get_setB_file_label(N):
	""" Returns set b file and label
		Input: N - minimum file length
	"""

	x_trainBFile = []  # wave filename
	y_trainB = []  # label
	framerate = []
	with open('Data/set_b.csv') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			if not row[2] == '' and not row[2] == 'label':
				fname = 'Data/set_b/' + row[1][21:]
				new_fname = ''
				count = 0
				for c in fname:
					new_fname += c
					if c == '_' and 'noisy' not in row[1]:
						if count == 1:
							new_fname += '_'
						count += 1
				if moreNsecs(new_fname, 2):
					x_trainBFile.append(new_fname)
					y_trainB.append(row[2])
					framerate.append(get_framerate(new_fname))
	return np.array(x_trainBFile), np.array(y_trainB), np.array(framerate)


def get_signal(name_of_file):
	""" Returns the signal from the file """
	f = wave.open(name_of_file)
	frames = f.readframes(-1)
	samples = struct.unpack('h'*f.getnframes(), frames)
	return samples


def get_raw_data(x_trainFile):
	""" Gets the raw x training data """
	raw_data = [np.array(get_signal(x_trainFile[i])) for i in range(len(x_trainFile))]
	return np.array(raw_data)


def down_sample(x, factor=10):
	""" Downsample one sample by a factor of N """

	r = range(len(x))
	down_array = [x[i] for i in r[0::factor]]
	return np.array(down_array)


def min_length(filenames):
	data = get_raw_data(filenames)
	return min(map(len, data))


def get_preprocessed_data(set_name, N=2, factor=10):
	"""Get preprocessing data and label for a set.
		Inputs:
			set_name - either A or B
			N - minimum file sound length (seconds)
			factor - downsample factor
	"""

	if set_name.upper() == 'A':
		filenames, y_label, framerates = get_setA_file_label(N)
	else:
		filenames, y_label, framerates = get_setB_file_label(N)
	raw_data = get_raw_data(filenames)  # get raw sound data
	x_data = np.array([down_sample(sample, factor=factor) for sample in raw_data])  # downsample
	x_data = np.array([pywt.dwt(x, 'db4')[0] for x in x_data])  # wavelet decomposition

	min_length = min(map(len, x_data))
	x_data = np.array([x[:min_length] for x in x_data])  # make all data the same length
	x_data = np.array([x/max(np.abs(x)) for x in x_data])  # normalize all data
	return x_data, y_label, framerates


def get_test_data(set_name, N=2, factor=10):

	if set_name.upper() == 'A':
		filenames = get_setA_testfile(N)
	else:
		return 'Currently not available'
	raw_data = get_raw_data(filenames)
	x_data = np.array([down_sample(sample, factor=factor) for sample in raw_data])  # downsample
	x_data = np.array([pywt.dwt(x, 'db4')[0] for x in x_data])  # wavelet decomposition
	x_data = np.array([x/max(np.abs(x)) for x in x_data])  # normalize all data
	return x_data
