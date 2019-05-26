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

name = ["Adele","Avril","BrunoMars","CheerChen","Eason","EdSheeran","JasonMraz","JJ","Ladygaga","TaylorSwift"] 

count = 0

for k in range(10):
  for i in range(1,11):

    file = "/home/username/new9/train/" + name[k] + "/" + str(i) +".wav"             # Set your path! 
    signal_all, fs = librosa.load(file,sr = 16000)

    for j in range(0,25):
  
      signal = signal_all[int(j * fs):int((j+1) * fs)]

      ####################librosa-mel#####################################################

      mel = librosa.feature.melspectrogram(y = signal, sr = fs, n_fft = 2048, hop_length = 512)

      mel = librosa.power_to_db(mel, ref = np.max)      #Convert scale to db
      """
      ####################librosa-mel-1#####################################################

      mel = librosa.feature.melspectrogram(y = signal, sr = fs, n_fft = 2048, hop_length = 512)

      
      mel = librosa.feature.delta(mel, order=1)         #This is for the derivative
      mel = librosa.power_to_db(mel, ref = np.max)      #Convert scale to db

      ####################librosa-mel-2#####################################################

      mel = librosa.feature.melspectrogram(y = signal, sr = fs, n_fft = 2048, hop_length = 512)

      
      mel = librosa.feature.delta(mel, order=2)         #This is for the derivative
      mel = librosa.power_to_db(mel, ref = np.max)      #Convert scale to db

      ####################power_spectrum#####################################################

      frames = speechpy.processing.stack_frames(signal, sampling_frequency=fs, frame_length=0.025, frame_stride=0.01, filter=lambda x: np.ones((x,)),zero_padding=True)


      power_spectrum = speechpy.processing.power_spectrum(frames, fft_points=512)


      #################### fft #####################################################
      frames = speechpy.processing.stack_frames(signal, sampling_frequency=fs, frame_length=0.025, frame_stride=0.01, filter=lambda x: np.ones((x,)),zero_padding=True)

      fft_spectrum = speechpy.processing.fft_spectrum(frames, fft_points = 512)

      #################### Merge mel and log-energy #####################################################


      mfcc = speechpy.feature.mfcc(signal, sampling_frequency=fs, frame_length=0.025, frame_stride=0.01,num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)

      logenergy = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=0.025, frame_stride=0.01, num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)

      merge_name = "/home/dl-linux/Desktop/new6/train/all/" + name[k] + "/" + str(i) + "_" + str(j) + ".jpg"

      merge = np.concatenate((mfcc,logenergy),axis = 1)
      """
      ####################save featrues#####################################################
      name = "/home/dl-linux/Desktop/new9/train/mel_2/" + name[k] + "/" + str(i) + "_" + str(j) + ".jpg"    # Set your path! 


      plt.imsave(name, mel)

      print(str(count) +  "/ 2500")



