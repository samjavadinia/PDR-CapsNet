# PDR-CapsNet
This is the code for Parallel Dynamic Routing Capsule Network (PDR-CapsNet) paper.

The following codes are used for this project:

https://github.com/pouyashiri/CFC-CapsNet.git

# Installation
Step 1. Install PyTorch and Torchvision:

conda install -c pytorch pytorch torchvision

Step 2. Install Torchnet:

pip install torchnet

# Usage

The "Main.py" file trains the network and prints the results to the files in the specified folder (input args).
Parameters:
--dset: Choice of dataset (options: MNIST, F-MNIST, SVHN and CIFAR-10)

--nc: Number of classes in the chosen dataset

--w : The width/height of input images

--bsize: Batch size

--ne: Number of epochs to train the model

--niter: Number of iterations for DR algorithm

--fck: Fully-Connected Kernel size (K parameter of the CFC layer)

--fdim: The output dimensionality (D parameter of the CFC layer)

--ich: number of channels in the input image

--dec_type: The type of decoder used (options: FC, DECONV)

--res_folder: The output folder to print the results into

--aug: Whether or not use a little augmentation to the dataset (options: 0,1)

--nc_recon: Performing the reconstruction in a single channel or all channels (options: 1,3)

--hard: Perform hard-training at the end or not (hard-training: training while tightening the bounds of the margin loss, options: 0,1)

--checkpoint: The file address of the checkpoint file (used for hard training)
