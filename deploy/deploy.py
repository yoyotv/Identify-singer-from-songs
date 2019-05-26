import caffe
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
import time



name = ["Adele","Avril","BrunoMars","CheerChen","Eason","EdSheeran","JasonMraz","JJ","Ladygaga","TaylorSwift"] 


save = np.full((25),-1)
judge = np.full((10),0)

for i in range(25):
  path = '/home/dl-linux/Desktop/test/mel/' + str(i) + '.jpg'
  img = mping.imread(path)
  #plt.imshow(img)
  #plt.show()

  img = np.swapaxes(img,0,1)
  img = np.swapaxes(img,0,2)
  img = np.expand_dims(img, axis = 0)

  #print(img.shape)
  #time.sleep(100)  

  caffe.set_mode_gpu()
  caffe.set_device(0)

  model = '/home/dl-linux/Desktop/new9/resnet_50_deploy.prototxt'
  weights = '/home/dl-linux/Desktop/new9/snapshot/mel_2_0.3/resnet_50_solver_iter_12429.caffemodel'



  net = caffe.Net(model, weights,caffe.TEST)

  net.blobs['data'].reshape(*img.shape)

  net.blobs['data'].data[0] = img

  determine = net.forward()

  #print(determine)
  #time.sleep(100)
  ans = str(np.argmax(determine['loss'][0]))
  ans_prob = str(determine['loss'][0][np.argmax(determine['loss'][0])])

  #print(name[int(ans)])
  
  #save[i] = name[int(ans)]
  save[i] = int(ans)

  #time.sleep(100)




for i in range(25):
  if save[i] == 0:
    judge[0] = judge[0] +1
  if save[i] == 1:
    judge[1] = judge[1] +1
  if save[i] == 2:
    judge[2] = judge[2] +1
  if save[i] == 3:
    judge[3] = judge[3] +1
  if save[i] == 4:
    judge[4] = judge[4] +1
  if save[i] == 5:
    judge[5] = judge[5] +1
  if save[i] == 6:
    judge[6] = judge[6] +1
  if save[i] == 7:
    judge[7] = judge[7] +1
  if save[i] == 8:
    judge[8] = judge[8] +1
  if save[i] == 9:
    judge[9] = judge[9] +1


print(name[np.argmax(judge)])

