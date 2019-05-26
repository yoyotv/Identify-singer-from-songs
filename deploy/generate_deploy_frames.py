import scipy.io.wavfile as wav
import numpy as np
import speechpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import librosa
import os
import time
import cv2
from PIL import Image



count = 0
#["Adele","Avril","BrunoMars","CheerChen","Eason","EdSheeran","JasonMraz","JJ","Ladygaga","TaylorSwift"] 

file = "/home/dl-linux/Desktop/Ed.wav"
signal_all, fs = librosa.load(file,sr = 16000)

for j in range(0,25):
  
      signal = signal_all[int(j * fs):int((j+1) * fs)]


      # Example of pre-emphasizing.
      signal_preemphasized = speechpy.processing.preemphasis(signal, cof=0.98)

      # Example of staching frames
      frames = speechpy.processing.stack_frames(signal, sampling_frequency=fs, frame_length=0.025, frame_stride=0.01, filter=lambda x: np.ones((x,)),zero_padding=True)



      mel = librosa.feature.melspectrogram(y = signal, sr = fs, n_fft = 2048, hop_length = 512)
      mel = librosa.feature.delta(mel, order=1)
      mel = librosa.power_to_db(mel, ref = np.max)



      """
# Example of staching frames
      frames = speechpy.processing.stack_frames(signal, sampling_frequency=fs, frame_length=0.025, frame_stride=0.01, filter=lambda x: np.ones((x,)),zero_padding=True)

            ############# Extract MFCC features #############
      mfcc = speechpy.feature.mfcc(signal, sampling_frequency=fs, frame_length=0.025, frame_stride=0.01,num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
      #mfcc_cmvn = speechpy.processing.cmvnw(mfcc,win_size=301,variance_normalization=True)
      #print('mfcc(mean + variance normalized) feature shape=', mfcc_cmvn.shape)

      mfcc_feature_cube = speechpy.feature.extract_derivative_feature(mfcc)
      #print('mfcc feature cube shape=', mfcc_feature_cube.shape)


      ############# Extract logenergy features #############
      logenergy = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=0.025, frame_stride=0.01, num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
      logenergy_feature_cube = speechpy.feature.extract_derivative_feature(logenergy)
      """




      mel_name = "/home/dl-linux/Desktop/test/mel/" + str(j) + ".jpg"




      count = count +1   

      plt.imsave(mel_name, mel)




