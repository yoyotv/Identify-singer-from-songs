import os
import numpy as np


name = ["Adele","Avril","BrunoMars","CheerChen","Eason","EdSheeran","JasonMraz","JJ","Ladygaga","TaylorSwift"] 





for i in range(10):
  for j in range(1,9):
    for k in range(0,25):
      with open("/home/dl-linux/Desktop/new9/train/mel_2/train.txt",'a') as file:
        file.write("/home/dl-linux/Desktop/new9/train/mel_2/" + name[i] + "/" + str(j) + "_" + str(k) + ".jpg" + " " + str(i) + "\n")


for i in range(10):
  for j in range(9,11):
    for k in range(0,25):
      with open("/home/dl-linux/Desktop/new9/train/mel_2/val.txt",'a') as file:
        file.write("/home/dl-linux/Desktop/new9/train/mel_2/" + name[i] + "/" + str(j) + "_" + str(k) + ".jpg" + " " + str(i) + "\n")



