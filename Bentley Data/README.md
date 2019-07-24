# Bentley Data

# What is in here?
In this folder, lies the several python files using the Data provided from Peter Bentley's challenge. There are several functions outlined in Audio Feature Extraction which help extract the audio, preprocessing (which preprocesses the data), and finally find features is the main script. In the misc folder, are a few helper functions designed to help organize the project. 

Note, most of this work has been done by [lindawangg](https://github.com/lindawangg/Classifying-Heartbeats) and I am adding to it by removing a few redundancies and speeding up the processes.

# Datasets

Download the [dataset](https://drive.google.com/open?id=1fIf_WYdc7Gu7PpYWM9BhxdsEtdt9a-Xt) in order to run the following script. The dataset has been structured a certain way, so extrat the entire folder Data and save it.

# Requirements

Python 3.6+

Numpy

# How to Run

To start, after downloading the data, run finding features. It will implement several ML algorithms---SVM, Naive Bayes, Linear Regression, and AdaBoost--- in an attempt to classiy the heart sounds. However, because his data was collected from a iPhone not a strong microphone in a controlled environment

# Results

So, from his dataset SVM performs the best and is the baseline any Deep Learning algorithm must beat.

# How it Works

Using Mel-frequency cepstral coefficients [(MFCCs)](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum), we transform all the normal and abnormal frequencies and then train the networks accordingly.

[Listen to Your Heart: Feature Extraction and Classification Methods for Heart Sounds](https://lindawangg.github.io/projects/classifying-heartbeats.pdf) outlines the process very smoothly.

# Helpful Links

[Linda Wang -- Classifying HeartBeats for Peter Bently Dataset](https://github.com/lindawangg/Classifying-Heartbeats)
