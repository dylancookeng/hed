import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
import scipy.misc
from PIL import Image, ImageSequence
import scipy.io
import scipy.ndimage
import os
from moviepy.editor import VideoFileClip
from scipy.ndimage import zoom

from pltimg import plot_list

caffe_root = '../../'
import sys
sys.path.insert(0,caffe_root+'python')

import caffe

data_root = '../../data/vids/'
video_name = 'vid1.mp4'
clip = VideoFileClip(data_root+video_name)

resize = False

if resize:
    for i, frame in enumerate(clip.iter_frames()):
        # print(frame.shape)
        frame = zoom(frame, (0.5,0.5,1))
        # print(frames[i].shape)

plt.ion()
ax = plt.subplot()
ax.set_title("HED LINE DETECTION")
ax.axis('off')
plt.pause(0.1)

for i, frame in enumerate(clip.iter_frames()):
    if i%15==0:

        in_ = frame
        #resize((400,400),Image.ANTIALIAS)
        in_ = in_.transpose((2,0,1))
        #remove the following two lines if testing with cpu
        caffe.set_mode_gpu()
        caffe.set_device(0)
        #load net
        model_root = './'
        # net = caffe.Net(model_root+'deploy.prototxt',model_root+'hed_pretrained_bsds.caffemodel', caffe.TEST)
        net = caffe.Net(model_root+'deploy.prototxt',model_root+'hed_iter_4000.caffemodel', caffe.TEST)
        # shape for input (data blob is NxCxHxW), set data
        net.blobs['data'].reshape(1,*in_.shape)
        net.blobs['data'].data[...] = in_
        # run net and take argmax for prediction
        net.forward()
        fuse = net.blobs['sigmoid-fuse'].data[0][0,:,:]
        result = frame.copy()
        for i in range(frame.shape[1]):
            for j in range(frame.shape[0]):
                if fuse[j,i] > 0.5:
                    result[j,i] = [255,0,0]
        img_plt = ax.imshow(result, cmap='gray')
        plt.pause(0.1)

plt.ioff()
plt.show()
