import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
import scipy.misc
from PIL import Image
import scipy.io
import os, time

from pltimg import plot_list

caffe_root = '../../'
import sys
sys.path.insert(0,caffe_root+'python')

import caffe

#data_root = '../../data/HED-BSDS/'
data_root = '../../data/trencher_follow_lines/'
with open(data_root+'test.lst') as f:
	test_lst = f.readlines()

test_lst = [data_root+x.strip() for x in test_lst]

im_lst = []
for i in range(0,len(test_lst)):
	im = Image.open(test_lst[i])
	# im = im.resize((400,400), Image.ANTIALIAS)
	in_ = np.array(im, dtype=np.float32)
	in_ = in_[:,:,::-1]
	in_ -= np.array((104.00698793,116.66876762,122.67891434))
	im_lst.append(in_)

for idx in range(len(test_lst)):
	
	if idx%3==0:
		continue

	start = time.time()
	in_ = im_lst[idx]
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
	out1 = net.blobs['sigmoid-dsn1'].data[0][0,:,:]
	out2 = net.blobs['sigmoid-dsn2'].data[0][0,:,:]
	out3 = net.blobs['sigmoid-dsn3'].data[0][0,:,:]
	out4 = net.blobs['sigmoid-dsn4'].data[0][0,:,:]
	out5 = net.blobs['sigmoid-dsn5'].data[0][0,:,:]
	fuse = net.blobs['sigmoid-fuse'].data[0][0,:,:]

	end = time.time()
	t = end-start
	print("Time {}".format(t))

	og = Image.open(test_lst[idx])
	result = og.copy()
	for i in range(og.size[1]):
		for j in range(og.size[0]):
			if fuse[i,j] > 0.5:
				result.putpixel((j,i),(255,0,0))
	
	scale_lst = [og,out1,out2,out3,out4,out5,fuse,result]
	fig2 = plot_list(scale_lst,'HED Side Outputs')
	plt.show()
	fig2.savefig('{}_HED.png'.format(test_lst[idx]))
	#plot_single_scale(scale_lst, 10, 'Side Outputs')
plt.show()