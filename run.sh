#!/bin/bash

for i in {1..5}
do

python main.py --dset cifar --nc 10 --w 32 --ich 3 --fdim 8 --res_folder lecaps --nc_recon 1 --hard 0
sh hardrun

python main.py --dset svhn --nc 10 --w 32 --ich 3 --fdim 8 --res_folder lecapse  --nc_recon 1 --hard 0
sh hardrun
##python main.py --dset mnist --nc 10 --w 28 --ich 1 --fdim 8 --res_folder ms-cfc --nc_recon 1 --hard 0
###sh hardrun
python main.py --dset norb --nc 5 --w 32 --ich 1 --fdim 8 --res_folder results --nc_recon 1 --hard 0
sh hardrun
#python main.py --dset mnist --nc 10 --w 28 --ich 1 --fdim 8 --res_folder results --nc_recon 1 --hard 0
#sh hardrun
# 


done