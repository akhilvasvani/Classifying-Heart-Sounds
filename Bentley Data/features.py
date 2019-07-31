from scipy.signal import find_peaks_cwt
import numpy as np


def find_peaks(samples, set_name):
	"""Gets a list of peaks for each sample"""
	if set_name.upper() == 'A':
		interval, r = 200, 5
	else:
		interval, r = 20, 2

	all_peaks = []
	for sample in samples:

		peaks = [i for i in find_peaks_cwt(sample, np.arange(1, r)) if sample[i] > 0.15]

		if len(peaks) > 1:
			start = 0
			# max index, and the peak value at the max index
			max_ind, max_peak = start, sample[peaks[start]]
			tmp_array = []

			for i in range(1, len(peaks)):
				if peaks[i] <= (peaks[start] + interval):
					if sample[peaks[i]] > max_peak:
						# set the max_index to where the current index is as well as the corresponding peak value
						max_ind, max_peak = i, sample[peaks[i]]
					if i == len(peaks)-1:
						tmp_array.append(peaks[max_ind])
				else:
					tmp_array.append(peaks[max_ind])
					start = i
					max_ind, max_peak = start, sample[peaks[start]]
		all_peaks.append(tmp_array)

	return np.array(all_peaks)
