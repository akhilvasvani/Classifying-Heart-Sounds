# Classifying-Heart-Sounds

# Purpose
This project is about developing a Deep Learning algorithm to correctly identity and classify heart sounds (specific to heart valve problems) into either an abnormal or normal cateogories. If the heart sound is classified is abnormal, what type of abnormality is it?

# Datasets

There are three types of datasets:

[Classifying Heart Sounds Challenge -- Peter Bentley](http://www.peterjbentley.com/heartchallenge/)

[Heart Sound & Murmur Library -- U Michigan](http://www.med.umich.edu/lrc/psb_open/html/repo/primer_heartsound/primer_heartsound.html)

[PhysioNet/CinC Challenge 2016: Training Sets](https://physionet.org/pn3/challenge/2016/)

# Requirements

Python 3.6+

TensorFlow 1.13+


# How to Run

To start, we use the dataset provided by Peter Bentley and implement several ML algorithms---SVM, Naive Bayes, Linear Regression, and AdaBoost--- in an attempt to classiy the heart sounds. Provided are two datasets: A & B. Dataset A is collected from an iPhone app and Dataset B is collected from patients in a hospitial. Using both datasets, SVM performs the best and is the baseline any Deep Learning algorithm must beat.

Next, using the dataset from U-Mich as well as others, developed four networks: a LSTM, a BiLSTM, a GRU, and a BiGRU network from the [paper](https://arxiv.org/pdf/1801.08322.pdf) to run and discover the accuracy. Compared with SVM, BiLSTM performs the best with an accuarcy of 98%.

# How it Works

Using Mel-frequency cepstral coefficients [(MFCCs)](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum), we transform all the normal and abnormal frequencies and then train the networks accordingly. Inside
the ``Bentley Data`` folder is the script for preprocessing the data, finding the features (using the ML methods), and audio feature extraction. Inside the ``models`` folder are the deep learning networks. 

# Helpful Links

[Linda Wang -- Classifying HeartBeats for Peter Bently Dataset](https://github.com/lindawangg/Classifying-Heartbeats)

[Classification of Heart Sound Signal Using Multiple Features](https://github.com/yaseen21khan/Classification-of-Heart-Sound-Signal-Using-Multiple-Features-)

[CNN model on Kaggle using Peter Bentley's Dataset](https://www.kaggle.com/kinguistics/heartbeat-sounds#set_b.csv)

[Heartbeat Sounds Kaggle](https://www.kaggle.com/kinguistics/heartbeat-sounds)

[Some implementations of RNN/ LSTM for Heart Sounds classification](https://github.com/vinayakumarr/Kalasalingam)



# Papers

[Listen to Your Heart: Feature Extraction and Classification Methods for Heart Sounds](https://lindawangg.github.io/projects/classifying-heartbeats.pdf) -- use SVM with Peter Bentley Dataset

[Recognizing Abnormal Heart Sounds Using Deep Learning](http://ceur-ws.org/Vol-1891/paper2.pdf)

[Phonocardiographic Sensing using Deep Learning for Abnormal Heartbeat Detection](https://arxiv.org/pdf/1801.08322.pdf) this!


# Notes
[Different spectrogram between audio_ops and tf.contrib.signal](https://stackoverflow.com/questions/53196156/different-spectrogram-between-audio-ops-and-tf-contrib-signal)
