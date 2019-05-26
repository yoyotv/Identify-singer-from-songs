# Identify-singer-from-songs

Sometimes we just want to know the singer of the current song that you are listening to. I train a CNN classification network to identify the singer by its feature. Trying fout kinds of different features, the best accuracy I could get is 39.2%. I think the scale of the training data is the main reason cause the low accuracy.

## Why

This repository is created because apply CNN in different fields are also challenging. This time I am trying to apply CNN in sound.

## Approach 

There are a lot of features can represent a sound. E.g., MEl, FFT, STFT. Most of them are focus on the frequency, After all, frequency means the band that a person speaks. **We try the mel, stft, log-energy, power spectrum , these four kinds of methods.**

Each song is about 30 seconds long. We take the first 25 seconds as our data. We will draw the spectrum in every 1 second. So we will have 25 frames of spectrum in one song. Totally we will have 25 (frames) * 10 (songs) * 10 (singers) = 2500 frames.

Seperate 2500 into 2000 (training) and 500 (testing). We choose the first 8 songs as the training data, the rest 2 songs as the testing data for every singer.**

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

## Start training!

1. Let's us librosa library to extract the mel features!

2. Download the [data](https://drive.google.com/open?id=1wSQbFb_mLSsVtb8iYHTJuOCQ7K5ANfTA)

3. Put the data under /Identify-singer-from-songs-master/train/, after that careate three folder named mel, mel_1, mel_2. It should look like this.

<img src="https://raw.githubusercontent.com/yoyotv/Identify-singer-from-songs/master/figures/1.JPG" >

5. Create folders like this in order to stroe the features. It should look like this.

<img src="https://raw.githubusercontent.com/yoyotv/Identify-singer-from-songs/master/figures/HI.JPG" >

4. Run ```python generate_features.py```.

5. Run ```python generate_list```. Don't forget to set the path in generate_list.

6. Copy your train.txt and test.txt to /Identify-singer-from-songs-master

7. Run caffe ```./build/tools/caffe train --solver=/home/username/Identify-singer-from-songs-master/resnet_50_solverprotxt```

## Notice 

1. If you are using the different diemnsion input, remember modify the settings in prototxt files.

## Tricks

1. **By trial and error, we found out that removing the 3x3 max pooling layer and modify the first convolutional layer from 7x7 to 5x5 will increase the accuracy about 10 %.**

2. If we replace the last global average pooling layer with fully-connected layer, the accuracy does not increase, sometimes it even get lower.

3. Dropout only increase accuracy about 1~2 %.
