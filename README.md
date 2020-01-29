# Classifying-Heart-Sounds

# Purpose
This project consists of developing a Deep Learning network to correctly identity and classify heart sounds (specific to heart valve problems) into either an abnormal or normal cateogories. If the heart sound is classified as abnormal, what type of abnormality is it?

# Datasets

There are three types of datasets:

[Classifying Heart Sounds Challenge -- Peter Bentley](http://www.peterjbentley.com/heartchallenge/)

[Heart Sound & Murmur Library -- U Michigan](http://www.med.umich.edu/lrc/psb_open/html/repo/primer_heartsound/primer_heartsound.html)

[PhysioNet/CinC Challenge 2016: Training Sets](https://physionet.org/pn3/challenge/2016/)

# Requirements

Python 3.6+

TensorFlow 1.13+

Keras 2.2.4+

librosa

scikit-learn


# How to Run

Inside the ``Bentley Data`` folder are the scripts for preprocessing the data, audio feature extraction, finding the features (using the ML algorithms---SVM, Naive Bayes, Linear Regression, and AdaBoost) to classify the heart sounds. Inside the ``models`` folder, which is inside the ``Bentley Data`` folder, are the Deep Learning networks---BiLSTM, LSTM, GRU, and BiGRU networks. To run any of the following scripts, 
clone the repo, and run each script. No arguments necessary.

# How it Works

To start, we use the dataset provided by Peter Bentley and implement several ML algorithms---SVM, Naive Bayes, Linear Regression, and AdaBoost--- in an attempt to classiy the heart sounds. Provided are two datasets of recorded heart sounds: A & B. Dataset A is collected from an iPhone app and Dataset B is collected from patients in a hospitial. 

One of the features of the hearts sounds is the Mel-frequency cepstral coefficients [(MFCCs)](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum), which is a representation of the short term power spectrum of a sound. Both the ML algorithms and Deep Learning methods learn to associate a specific MFCC with a category (in this case, abnormal or normal heartbeat). Using both datasets, SVM performs the best amongst the ML algorithms and is the baseline any Deep Learning network must beat.

Next there are four Deep Learning networks: a LSTM, a BiLSTM, a GRU, and a BiGRU network adapted from the [paper](https://arxiv.org/pdf/1801.08322.pdf). These networks were primarily used because LSTMs and GRUs are both capable of learning long-term dependencies. In simple terms, these networks are able to connect previous information to the present task without consuming too much memory. Compared with SVM, BiLSTM performs the best with an accuarcy of 98%; however, I suspect that the models are overfitting the data.


# Things to Potentially Improve

1. Find more data

2. Change the number of MFCCs (right now, I use 40 but is the correct number?)

3. Fiddle with the number of units in the LSTM cell (I have 64 units for the first cell, and 32 for the next)

4. Change the optimizer (I use Adam, but should I consider SGD?)


# Helpful Links

[Linda Wang -- Classifying HeartBeats for Peter Bently Dataset](https://github.com/lindawangg/Classifying-Heartbeats)

[Classification of Heart Sound Signal Using Multiple Features](https://github.com/yaseen21khan/Classification-of-Heart-Sound-Signal-Using-Multiple-Features-)

[CNN model on Kaggle using Peter Bentley's Dataset](https://www.kaggle.com/kinguistics/heartbeat-sounds#set_b.csv)

[Heartbeat Sounds Kaggle](https://www.kaggle.com/kinguistics/heartbeat-sounds)

[Some implementations of RNN/ LSTM for Heart Sounds classification](https://github.com/vinayakumarr/Kalasalingam)



# Papers

[Listen to Your Heart: Feature Extraction and Classification Methods for Heart Sounds](https://lindawangg.github.io/projects/classifying-heartbeats.pdf) -- use SVM with Peter Bentley Dataset

[Recognizing Abnormal Heart Sounds Using Deep Learning](http://ceur-ws.org/Vol-1891/paper2.pdf)

[Phonocardiographic Sensing using Deep Learning for Abnormal Heartbeat Detection](https://arxiv.org/pdf/1801.08322.pdf)


# Notes

[Different spectrogram between audio_ops and tf.contrib.signal](https://stackoverflow.com/questions/53196156/different-spectrogram-between-audio-ops-and-tf-contrib-signal)



# References

Thank you [Vivek Karn](https://github.com/vivekkarn/classification-of-heart-sounds) for you initial repo, which helped give me an idea of how to start.
Thank you [Vinaya Kumar](https://github.com/vinayakumarr/Kalasalingam) for your repo and various papers included.
