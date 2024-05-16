**Holistically Nested Edge Detection**
==============================


Forked from [GitHub - s9xie/hed: code for Holistically-Nested Edge Detection](https://github.com/s9xie/hed)

## Build Instructions
### CPU-Only Version with Python Overlay

#### Prerequisites

* Ubuntu system (WSL with Ubuntu 22.04)
	+ Run `wsl --install` in Admin CMD
* Clone the repo
	+ `git clone https://github.com/dylancookeng/hed.git`
	+ `cd hed`

#### Install Dependencies

* Get g++ and gcc
	+ `sudo apt install g++ gcc`
	+ `sudo apt install make`
* Get Python 2.7 and dev libs
	+ `sudo apt install python2`
	+ `sudo apt install python2-dev`
* Get pip for Python 2.7
	+ `curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py`
	+ `python2 get-pip.py`
	+ `export PATH="$HOME/.local/bin:$PATH"`

#### Install All Python Dependencies

* `cd python`
* `for req in $(cat requirements.txt); do pip install $req; done`
* `sudo apt install python-tk`
* `pip install matplotlib pillow`

#### Install BLAS, Protobuf, and Glog/Gflags/HDF5

* `sudo apt install libblas3`
* `sudo apt-get install libatlas-base-dev liblapack-dev libblas-dev`
* `sudo apt install libprotobuf-dev protobuf-compiler`
* `sudo apt install libgoogle-glog-dev libgflags-dev libhdf5-dev`

#### Install IO Libs

* `sudo apt install liblmdb-dev libleveldb-dev`

#### Install Snappy

* `sudo apt install libsnappy-dev`

#### Build Old Version of Boost (1.64.0)

* `cd /usr/local`
* `sudo curl -L https://boostorg.jfrog.io/artifactory/main/release/1.64.0/source/boost_1_64_0.tar.bz2 --output boost_1_64_0.tar.bz2`
* `sudo tar -bzip2 -xf ./boost_1_64_0.tar.bz2`
* `cd boost_1_64_0`
* `sudo ./bootstrap.sh --prefix='/usr/local/boost_1_64_0/mybuild' --with-libraries=python`
* `sudo ./b2 --build-dir=/tmp/build-boost --prefix='/usr/local/boost_1_64_0/mybuild' --with-python`

#### Add Boost Libraries

* `sudo nano /etc/ld.soconf.d/boost_1_64_0.conf`
  * Add `/usr/local/boost_1_64_0/stage/lib`
  * Save and exit `ctrl+x` `y` `enter`
* `sudo ldconfig`

#### Build OpenCV 2

* `sudo apt install cmake`
* `cd ~`
* `git clone https://github.com/opencv/opencv.git`
* `cd opencv`
* `git checkout 2.4`
* `mkdir -p build && cd build`
* `cmake ../`
* `cmake --build .`
* `sudo make install`

#### Configure the HED repo for building

* `cd ~/hed`
* `cp Makefile.config.example Makefile.config`
* Edit Makefile.config with nano or editor of choice
  * `nano Makefile.config`
  * Line 8: Uncomment `CPU_ONLY:=1`
  * Line 69: Uncomment `WITH_PYTHON_LAYER:=1`
  * Line 71: Add `/usr/local/boost_1_64_0`
  * Line 72: Add `/usr/include/hdf5/serial /usr/local/boost_1_64_0/stage/lib`
* Edit Makefile
  * `nano Makefile`
  * Line 173: rename `hdf5_hl` and `hdf5` to `hdf5_serial_hl` and `hdf5_serial`, respectively

#### Build HED

* `make all -j8`
* `make test -j8`
* `make runtest`
  * Some of the tests will fail because it does not have all the data it needs.

#### Build Python Interface

* For Ubuntu 20.04
  * `sudo apt install python-numpy`
* For Ubuntu 22.04
  * `pip2 install numpy`
  * `sudo ln -s $HOME/.local/python2.7/site-packages/numpy/core/include/numpy/ /usr/include/numpy`
* `make pycaffe`

#### Download Models and Data
* Pre-trained model
  * `curl -L https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel --output ./examples/hed/hed_pretrained_bsds.caffemodel`
* Un-trained model
  * `curl -L https://vcl.ucsd.edu/hed/5stage-vgg.caffemodel --output ./examples/hed/5stage-vgg.caffemodel`
* Training data
  * `mkdir data`
  * `cd data`
  * `curl -L https://vcl.ucsd.edu/hed/HED-BSDS.tar --output HED-BSDS.tar`
  * `tar -xf HED-BSDS.tar`

### For GPU...

* Follow cpu instructions first, then continue here.

#### Install Cuda (Caffe calls for 7+, lets try 12.4)

* `wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin`
* `sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600`
* `wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-wsl-ubuntu-12-4-local_12.4.1-1_amd64.deb`
* `sudo dpkg -i cuda-repo-wsl-ubuntu-12-4-local_12.4.1-1_amd64.deb`
* `sudo cp /var/cuda-repo-wsl-ubuntu-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/`
* `sudo apt-get update`
* `sudo apt-get -y install cuda-toolkit-12-4`
* `rm cuda-repo-wsl-ubuntu-12-4-local_12.4.1-1_amd64.deb`

#### Install Cudnn

* `wget https://developer.download.nvidia.com/compute/cudnn/9.1.0/local_installers/cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.deb`
* `sudo dpkg -i cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.deb`
* `sudo cp /var/cudnn-local-repo-ubuntu2204-9.1.0/cudnn-*-keyring.gpg /usr/share/keyrings/`
* `sudo apt-get update`
* `sudo apt-get -y install cudnn`
* `rm cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.deb`

#### Edit Makefile.config

* Line 8: comment out `CPU_ONLY:=1`
* Line 22: change to the following -
  * CUDA_ARCH := -gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_50,code=compute_50 \
		-gencode arch=compute_60,code=sm_60 \
		-gencode arch=compute_61,code=sm_61 \
		-gencode arch=compute_61,code=compute_61

#### Build HED

* `make all -j8`
* `make test -j8`
* `make runtest`
  * Some of the tests will fail because it does not have all the data it needs.


#### Using HED
* For training and testing the HED algorithm, see the original README below.


## Holistically-Nested Edge Detection

Created by Saining Xie at UC San Diego

### Introduction:

<img src="http://pages.ucsd.edu/~ztu/hed.jpg" width="400">

We develop a new edge detection algorithm, holistically-nested edge detection (HED), which performs image-to-image prediction by means of a deep learning model that leverages fully convolutional neural networks and deeply-supervised nets.  HED automatically learns rich hierarchical representations (guided by deep supervision on side responses) that are important in order to resolve the challenging ambiguity in edge and object boundary detection. We significantly advance the state-of-the-art on the BSD500 dataset (ODS F-score of .790) and the NYU Depth dataset (ODS F-score of .746), and do so with an improved speed (0.4s per image). Detailed description of the system can be found in our [paper](http://arxiv.org/abs/1504.06375).

### Citations

If you are using the code/model/data provided here in a publication, please cite our paper:

    @InProceedings{xie15hed,
      author = {"Xie, Saining and Tu, Zhuowen"},
      Title = {Holistically-Nested Edge Detection},
      Booktitle = "Proceedings of IEEE International Conference on Computer Vision",
      Year  = {2015},
    }

### Changelog

If you have downloaded the previous version (testing code) of HED, please note that we updated the code base to the new version of Caffe. We uploaded a new pretrained model with better performance. We adopted the python interface written for the FCN paper instead of our own implementation for training and testing. The evaluation protocol doesn't change.

### Pretrained model

We provide the pretrained model and training/testing code for the edge detection framework Holistically-Nested Edge Detection (HED). Please see the Arxiv or ICCV paper for technical details. The pretrained model (fusion-output) gives ODS=.790 and OIS=.808 result on BSDS benchmark dataset.
  0. Download the pretrained model (56MB) from (https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel) and place it in examples/hed/ folder.

### Installing 
 0. Install prerequisites for Caffe(http://caffe.berkeleyvision.org/installation.html#prequequisites)
 0. Modified-caffe for HED: https://github.com/s9xie/hed.git

### Training HED
To reproduce our results on BSDS500 dataset:
 0. data: Download the augmented BSDS data (1.2GB) from (https://vcl.ucsd.edu/hed/HED-BSDS.tar) and extract it in data/ folder
 0. initial model: Download fully convolutional VGG model (248MB) from (https://vcl.ucsd.edu/hed/5stage-vgg.caffemodel) and put it in examples/hed folder
 0. run the python script **python solve.py** in examples/hed

### Testing HED
Please refer to the IPython Notebook in examples/hed/ to test a trained model. The fusion-output, and individual side-output from 5 scales will be produced after one forward pass.
 
Note that if you want to evaluate the results on BSDS benchmarking dataset, you should do the standard non-maximum suppression (NMS) and edge thinning. We used Piotr's Structured Forest matlab toolbox available here **https://github.com/pdollar/edges**. Some helper functions are also provided in the [eval/ folder](https://github.com/s9xie/hed_release-deprecated/tree/master/examples/eval). 

### Batch Processing

[Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz/) from UC Berkeley recently applied HED for their [Image-to-Image Translation](https://phillipi.github.io/pix2pix/) work. A nice script for batch-processing HED edge detection can be found [here](https://github.com/phillipi/pix2pix/tree/master/scripts/edges). Thanks Jun-Yan!

### Precomputed Results
If you want to compare your method with HED and need the precomputed results, you can download them from (https://vcl.ucsd.edu/hed/eval_results.tar).


### Acknowledgment: 
This code is based on Caffe. Thanks to the contributors of Caffe. Thanks @shelhamer and @longjon for providing fundamental implementations that enable fully convolutional training/testing in Caffe.

    @misc{Jia13caffe,
      Author = {Yangqing Jia},
      Title = { {Caffe}: An Open Source Convolutional Architecture for Fast Feature Embedding},
      Year  = {2013},
      Howpublished = {\url{http://caffe.berkeleyvision.org/}}
    }

If you encounter any issue when using our code or model, please let me know.
