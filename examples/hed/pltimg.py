import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
import scipy.misc
from PIL import Image
import scipy.io
import os

def plot_list(imgs, title, size=(12,4), single=False):
	fig, axs = plt.subplots(1, len(imgs), figsize=size)
	if single:
		axs.imshow(imgs[0], cmap='gray')
		axs.set_xticklabels([])
		axs.set_yticklabels([])
		axs.yaxis.set_ticks_position('none')
		axs.xaxis.set_ticks_position('none')
	else:
		for i in range(0, len(imgs)):
			axs[i].imshow(imgs[i], cmap='gray')
			axs[i].set_title('{}'.format(i))
			axs[i].set_xticklabels([])
			axs[i].set_yticklabels([])
			axs[i].yaxis.set_ticks_position('none')
			axs[i].xaxis.set_ticks_position('none')
	fig.suptitle(title)
	return fig

if __name__ == '__main__':
	img1 = '../../data/report/images.jpg'
	img2 = '../../data/report/IMG_7909.jpeg'
	imgs = [Image.open(img1), Image.open(img2)]
	plot_list(imgs, 'testing')
	plt.show()
