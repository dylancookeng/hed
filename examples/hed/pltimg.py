import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
import scipy.misc
from PIL import Image
import scipy.io
import os

def plot_list(imgs, title, size=(16,8), single=False):
	fig, axs = plt.subplots(2, len(imgs)//2 + (len(imgs) % 2 != 0), figsize=size)
	if single:
		axs.imshow(imgs[0], cmap='gray')
		axs.set_xticklabels([])
		axs.set_yticklabels([])
		axs.yaxis.set_ticks_position('none')
		axs.xaxis.set_ticks_position('none')
	else:
		for i, ax in enumerate(axs.flat):
			if i >= len(imgs):
				break
			ax.imshow(imgs[i], cmap='gray')
			ax.set_title('{}'.format(i))
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.yaxis.set_ticks_position('none')
			ax.xaxis.set_ticks_position('none')
	fig.suptitle(title)
	return fig

if __name__ == '__main__':
	img1 = '../../data/report/images.jpg'
	img2 = '../../data/report/IMG_7909.jpeg'
	imgs = [Image.open(img1), Image.open(img2)]
	plot_list(imgs, 'testing')
	plt.show()
