# Identify-singer-from-songs

Sometimes we just want to know the singer of the current song that you are listening to. I train a CNN classification network to identify the singer by its feature. Trying fout kinds of different features, the best accuracy I could get is 39.2%. I think the scale of the training data is the main reason cause the low accuracy.

## Why

This repository is created because apply CNN in different fields are also challenging. This time I am trying to apply CNN in sound.

## Approach 

There are a lot of features can represent a sound. E.g., MEl, FFT, STFT. Most of them are focus on the frequency, After all, frequency means the band that a person speaks. **We try the mel, stft, log-energy, power spectrum , these four kinds of methods.**

We use modified-Resnet-50 as our CNN backbone.

## GET started

You have to install

1. Python ( I use 2.7 )

2. [Caffe](https://github.com/BVLC/caffe) for training and classify.

3. [Speechpy](https://github.com/astorfi/speechpy) for extracting features.

4. [Librosa](https://librosa.github.io/librosa/) for extracting features.

## Installation

1. Clone the repository
  ```
  git clone https://github.com/yoyotv/Identify-singer-from-songs.git
  ```

2. Finished!

## Tutorial

**The folder name is base on the different feature that We use for training. Let us use librosa-mel as our tutorial.**

Assume we are under librosa-mel.

1. ```cd train  ```

2. ```python generate_list.py``` (Don't forget change the path in generate_list.py !)

3. ```cd mel``` and copy the train.txt and val.txt to /librosa-mel

4. Run caffe ```./build/tools/caffe train --solver=/home/username/librosa-mel/resnet_50_solverprotxt```




1. 
