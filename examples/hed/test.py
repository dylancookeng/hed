import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
import scipy.misc
from PIL import Image
import scipy.io
import os

from pltimg import plot_list

caffe_root = '../../'
import sys
sys.path.insert(0,caffe_root+'python')

import caffe

#data_root = '../../data/HED-BSDS/'
data_root = '../../data/report/'
with open(data_root+'test.lst') as f:
	test_lst = f.readlines()

test_lst = [data_root+x.strip() for x in test_lst]

im_lst = []
for i in range(0,len(test_lst)):
	im = Image.open(test_lst[i])
	in_ = np.array(im, dtype=np.float32)
	in_ = in_[:,:,::-1]
	in_ -= np.array((104.00698793,116.66876762,122.67891434))
	im_lst.append(in_)

#Visualization
def plot_single_scale(scale_lst, size, title):
	pylab.rcParams['figure.figsize'] = size, size/2
	
	plt.figure()
	for i in range(0,len(scale_lst)):
		s=plt.subplot(1,5,i+1)
		plt.imshow(scale_lst[i], cmap='gray')
		s.set_xticklabels([])
		s.set_yticklabels([])
		s.yaxis.set_ticks_position('none')
		s.xaxis.set_ticks_position('none')
	plt.tight_layout()
	plt.suptitle(title)

for idx in range(len(test_lst)):
	in_ = im_lst[idx]
	in_ = in_.transpose((2,0,1))
	#remove the following two lines if testing with cpu
	#caffe.set_mode_gpu()
	#caffe.set_device(0)
	#load net
	model_root = './'
	net = caffe.Net(model_root+'deploy.prototxt',model_root+'hed_pretrained_bsds.caffemodel', caffe.TEST)
	# shape for input (data blob is NxCxHxW), set data
	net.blobs['data'].reshape(1,*in_.shape)
	net.blobs['data'].data[...] = in_
	# run net and take argmax for prediction
	net.forward()
	out1 = net.blobs['sigmoid-dsn1'].data[0][0,:,:]
	out2 = net.blobs['sigmoid-dsn2'].data[0][0,:,:]
	out3 = net.blobs['sigmoid-dsn3'].data[0][0,:,:]
	out4 = net.blobs['sigmoid-dsn4'].data[0][0,:,:]
	out5 = net.blobs['sigmoid-dsn5'].data[0][0,:,:]
	fuse = net.blobs['sigmoid-fuse'].data[0][0,:,:]

	scale_lst = [fuse]
	#print(scale_lst)
	#plot_single_scale(scale_lst, 22, 'Final Result')
	fig1 = plot_list(scale_lst,'HED Final Result',single=True)
	fig1.savefig('img{}_Final_Result.png'.format(idx))
	scale_lst = [out1,out2,out3,out4,out5]
	fig2 = plot_list(scale_lst,'HED Side Outputs')
	fig2.savefig('img{}_Side_Outputs.png'.format(idx))
	#plot_single_scale(scale_lst, 10, 'Side Outputs')
plt.show()
