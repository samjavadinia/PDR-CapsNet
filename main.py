"""
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829

PyTorch implementation by Kenta Iwasaki @ Gram.AI.
Code Enhancement by Samaneh Javadinia sa.javadinia@gmail.com
"""
import sys, os
sys.setrecursionlimit(15000)
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np, time
import pickle
from torch.autograd import Variable
from torch.optim import Adam
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from torchvision.utils import make_grid

from tqdm import tqdm
import torchnet as tnt
import argparse
from dataset_loader import *


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


def augmentation(x, max_shift=2):
    _, _, height, width = x.size()

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()


class FlattenLayer(nn.Module):
    def __init__(self, D_in, C_in, D_out, kernel):
        super(FlattenLayer, self).__init__()
        self.kernel = kernel
        self.D_in = D_in
        self.D_out = D_out
        self.num_fc_sq = (D_in - kernel + 1)

    def forward(self, x):
        out_fc = []
        num_ks = self.num_fc_sq
        for ix in range(num_ks):
            for iy in range(num_ks):
                x_part = x[:, :, ix:ix + self.kernel, iy:iy + self.kernel]
                x_part = nn.Flatten()(x_part)
                x_part = x_part.unsqueeze(1)
                out_fc.append(x_part)

        return torch.cat(out_fc, dim=1)






class CFCLayer(nn.Module):
    def __init__(self, D_in, C_in, D_out, kernel):
        super(CFCLayer, self).__init__()
        self.kernel = kernel
        self.D_in = D_in
        self.D_out = D_out
        self.num_fc_sq = (D_in - kernel + 1)
        self.FCs = nn.ModuleList([nn.Linear(kernel * kernel * C_in, D_out) for _ in range(self.num_fc_sq ** 2)])

    def forward(self, x):
        out_fc = []
        num_ks = self.num_fc_sq
        for ix in range(num_ks):
            for iy in range(num_ks):
                x_part = x[:, :, ix:ix + self.kernel, iy:iy + self.kernel]
                x_part = nn.Flatten()(x_part)
                x_part = self.FCs[ix * num_ks + iy](x_part).unsqueeze(1)
                out_fc.append(x_part)
                
        
        return torch.cat(out_fc, dim=1)


class fc_1d(nn.Module):
    def __init__(self, D_in, C_in, D_out, kernel):
        super(fc_1d, self).__init__()
        self.kernel = kernel
        self.D_in = D_in
        self.D_out = D_out
        self.C_in=C_in
        self.num_fc_sq = (D_in - kernel + 1)
        self.FCs =nn.Linear(C_in,D_out)

    def forward(self, x):
        out_fc = []
        num_ks = (self.num_fc_sq)
     
        
        for ix in range(num_ks):
    
            x_part = x[:, ix:ix + self.kernel, :]
            x_part = nn.Flatten()(x_part)
            #print('x_part===',x_part.size())
            x_part = self.FCs(x_part).unsqueeze(1)
            #print('x_part===',x_part.size())
            out_fc.append(x_part)
        #print('XSIZE in FCLAYER',x_part.size())
        return torch.cat(out_fc, dim=1)








"""
Randomly drops some capsules during training (Inverse Dropout)
Inputs:
drp_rate (float): the drop rate
is_training (boolean): True for training, False otherwise
x (Tensor): (None, num_capsules, capsule_dims)  
"""
class CapsuleDropout(nn.Module):
    def __init__(self, drp_rate):
        super(CapsuleDropout, self).__init__()
        self.drp_rate = drp_rate


    def forward(self, x, is_training):
        if self.drp_rate == 0 or is_training == False:
            return x

        # # during testing, the tensor is multiplied by dropout rate
        # if is_training == False:
        #     return x*self.drp_rate

        mask = np.random.binomial(1, 1 - self.drp_rate, x.size(0) * x.size(1)).reshape((x.size(0), x.size(1)))
        maskTensor = torch.tensor(mask)
        maskTensor = torch.unsqueeze(maskTensor, -1)
        maskTensor = maskTensor.float().cuda()
        return (x * maskTensor)/(1-self.drp_rate)

"""
The decoder of CapsNet, reconstructs the input images using the capsule encoding. 
Inputs:
w: width and height of the final conv feature map in the encoder
org_w: width and height of the images
nc: number of classes
nc_recon: number of reconstructed channels (grayscale reconstruction works better)
type: the type of the decoder (FC, DECONV)
NOTE: currently only supported for 32x32 and 28x28 images
"""
class CapsNetDecoder(nn.Module):
    def __init__(self, org_w, nc, nc_recon, type):
        super(CapsNetDecoder, self).__init__()
        self.type = type
        self.w = 10 if org_w == 32 else 6
        self.nc = nc

        if type == 'FC':
            # updated decoder to get both output vectors
            self.decoder = nn.Sequential(
                nn.Linear(16 * nc, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, org_w ** 2 * nc_recon),
                nn.Sigmoid()
            )
        else:
            self.fc_end = nn.Linear(16, 8 * (self.w ** 2))
            self.bn = nn.BatchNorm1d(8 * (self.w ** 2), momentum=0.8)
            self.deconvs = nn.Sequential(
                nn.ConvTranspose2d(8, 128, 3),
                nn.ConvTranspose2d(128, 64, 5),
                nn.ConvTranspose2d(64, 32, 5),
                nn.ConvTranspose2d(32, 16, 5),
                nn.ConvTranspose2d(16, 16, 5),
                nn.ConvTranspose2d(16, 16, 3),
                nn.ConvTranspose2d(16, nc_recon, 3))

    def forward(self, x, y, classes):
        if self.type == 'FC':
            if y is None:
                # In all batches, get the most active capsule.
                _, max_length_indices = classes.max(dim=1)
                y = Variable(torch.eye(self.nc)).cuda().index_select(dim=0, index=max_length_indices.data)
            reconstructions = self.decoder((x * y[:, :, None]).reshape(x.size(0), -1))
        else:
            if y is None:
                _, max_length_indices = classes.max(dim=1)
            else:
                _, max_length_indices = y.max(dim=1)

            reconRes = torch.Tensor(np.zeros((x.size(0), x.size(2)))).cuda()
            for i in range(x.size(0)):
                reconRes[i, :] = x[i, max_length_indices[i], :]

            x = self.bn(self.fc_end(reconRes))

            x = x.reshape(x.size(0), 8, self.w, self.w)

            x = F.relu(self.deconvs(x))
            reconstructions = x.reshape(x.size(0), -1)

        return classes, reconstructions


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, in_channels, out_channels, num_classes,
                 num_iterations=3):
        super(CapsuleLayer, self).__init__()
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules
        self.num_classes = num_classes
        self.route_weights = nn.Parameter(torch.randn(num_classes, num_capsules, in_channels, out_channels))

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):

        # print(f'###{x[None, :, :, None, :].size()}-{self.route_weights[:, None, :, :, :].size()}')
        priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

        logits = Variable(torch.zeros(*priors.size())).cuda()
        for i in range(self.num_iterations):
            probs = softmax(logits, dim=2)
            outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

            if i != self.num_iterations - 1:
                delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                logits = logits + delta_logits

        return outputs


class CapsuleNet(nn.Module):
    def __init__(self, num_class, niter, width, in_channels=1, fc_kernel_size=1, fc_out_dim=8, drp=0, nc_recon=1,
                 decoder_type='FC'):
        super(CapsuleNet, self).__init__()

        self.decoder_type = decoder_type
        self.nc_recon = nc_recon
        self.drp = drp
        self.nc = num_class


        """
        implementing Conv without routing paper

        """
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(16)



        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.conv1_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.bn1_3 = nn.BatchNorm2d(32)

        self.conv1_4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.bn1_4 = nn.BatchNorm2d(64)
        self.conv1_5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.bn1_5 = nn.BatchNorm2d(64)

        self.conv1_6 = nn.Conv2d(in_channels=64, out_channels=80, kernel_size=3)
        self.bn1_6 = nn.BatchNorm2d(80)
        self.conv1_7 = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3)
        self.bn1_7 = nn.BatchNorm2d(80)

        self.conv1_8 = nn.Conv2d(in_channels=80, out_channels=96, kernel_size=3)
        self.bn1_8 = nn.BatchNorm2d(96)
        self.conv1_9 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3)
        self.bn1_9 = nn.BatchNorm2d(96)




        self.depthconv2=nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3,groups=96)
        self.pointconv2=nn.Conv2d(in_channels=96, out_channels=112, kernel_size=1)
        self.depthwise_separable_conv2 = torch.nn.Sequential(self.depthconv2, self.pointconv2)
        # self.conv2 = nn.Conv2d(in_channels=96, out_channels=112, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(112)
        # self.conv3 = nn.Conv2d(in_channels=112, out_channels=160, kernel_size=3)
        self.depthconv3=nn.Conv2d(in_channels=112, out_channels=112, kernel_size=3,groups=112)
        self.pointconv3=nn.Conv2d(in_channels=112, out_channels=128, kernel_size=1)
        self.depthwise_separable_conv3 = torch.nn.Sequential(self.depthconv3, self.pointconv3)
        self.bn3 = nn.BatchNorm2d(128)

        self.depthconv4=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,groups=128)
        self.pointconv4=nn.Conv2d(in_channels=128, out_channels=140, kernel_size=1)
        self.depthwise_separable_conv4 = torch.nn.Sequential(self.depthconv4, self.pointconv4)
        self.bn4 = nn.BatchNorm2d(140)
        self.depthconv5=nn.Conv2d(in_channels=140, out_channels=140, kernel_size=3,groups=140)
        self.pointconv5=nn.Conv2d(in_channels=140, out_channels=156, kernel_size=1)
        self.depthwise_separable_conv5 = torch.nn.Sequential(self.depthconv5, self.pointconv5)
        self.bn5 = nn.BatchNorm2d(156)
        self.depthconv6=nn.Conv2d(in_channels=156, out_channels=156, kernel_size=3,groups=156)
        self.pointconv6=nn.Conv2d(in_channels=156, out_channels=172, kernel_size=1)
        self.depthwise_separable_conv6 = torch.nn.Sequential(self.depthconv6, self.pointconv6)
        self.bn6 = nn.BatchNorm2d(172)











        self.width = width
        w1=width
        
        for i in range (8):
            w1 = w1 - 3 + 1

        w1 = w1 - 3 + 1
        w2 = w1 - 3 + 1
        w2 = w2 - 3 + 1
        w3 = w2 - 3 + 1
        w3 = w3 - 3 + 1

        print('W1,w2,w3',w1,w2,w3)


        # for i in range (3):
            #  w1 = w1 - 3 + 1

        # for i in range (6):
            #  w2 = w2 - 3 + 1




        # print('W11111===',w1)
        self.cfc1 = CFCLayer(w1, 112, 8, fc_kernel_size)
        
        self.cfc2 = CFCLayer(w2, 140, 8, fc_kernel_size)

        self.cfc3 = CFCLayer(w3, 172, 8, fc_kernel_size)
        # w1 = w1 - 2 
        # w2 = w2 - 2
        # w3 = w3 - 2 





        self.fc_out_dim = fc_out_dim
        
        fc_sq_1 = ( w1 ** 2 )
        print('############fc_sq_1===',fc_sq_1)


        fc_sq_2 = ( w2 ** 2 )
        print('############fc_sq_2===',fc_sq_2)

 
        fc_sq_3 = ( w3 ** 2 )
        print('############fc_sq_3===',fc_sq_3)
    

        self.caps1 = CapsuleLayer(num_capsules=(fc_sq_1), in_channels=8, out_channels=16,
                                  num_classes=num_class,
                                  num_iterations=niter)



        self.caps2 = CapsuleLayer(num_capsules=(fc_sq_2), in_channels=8, out_channels=16,
                                  num_classes=num_class,
                                  num_iterations=niter)



        self.caps3 = CapsuleLayer(num_capsules=(fc_sq_3), in_channels=8, out_channels=16,
                                  num_classes=num_class,
                                  num_iterations=niter)

                          

        self.dropout = CapsuleDropout(drp)
      
        self.decoder = CapsNetDecoder(width, num_class, nc_recon, decoder_type)



    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x, y=None):
        x = x.float()
        # print('X1=====     startttttttttttt',x)
        # print('X1=====     finishhhhhhhhhhh')

        #x = F.relu(self.conv0(x), inplace=True)
        x1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x1 = F.relu(self.bn1_2(self.conv1_2(x1)), inplace=True)
        x1 = F.relu(self.bn1_3(self.conv1_3(x1)), inplace=True)
        x1 = F.relu(self.bn1_4(self.conv1_4(x1)), inplace=True)
        x1 = F.relu(self.bn1_5(self.conv1_5(x1)), inplace=True)
        x1 = F.relu(self.bn1_6(self.conv1_6(x1)), inplace=True)
        x1 = F.relu(self.bn1_7(self.conv1_7(x1)), inplace=True)
        x1 = F.relu(self.bn1_8(self.conv1_8(x1)), inplace=True)
        x1 = F.relu(self.bn1_9(self.conv1_9(x1)), inplace=True)





        x1 = self.bn2(self.depthwise_separable_conv2(x1))
        x1_1 = F.relu((x1), inplace=True)

        # print('X1=====     startttttttttttt',x1.size())
        # print('X1=====     finishhhhhhhhhhh')

        x2 = F.relu(self.bn3(self.depthwise_separable_conv3(x1_1)), inplace=True)
        x2 = self.bn4(self.depthwise_separable_conv4(x2))
        x2_1 = F.relu((x2), inplace=True)

        # print('X2=====',x2.size())
 
        x3 = F.relu(self.bn5(self.depthwise_separable_conv5(x2_1)), inplace=True)
        x3 = self.bn6(self.depthwise_separable_conv6(x3))

        # print('X3=====',x3.size())



        cfc1 =self.squash( self.cfc1(x1), dim=-1)
        # print('CFC1=====',cfc1.size())


        cfc2 =self.squash( self.cfc2(x2), dim=-1)
        # print('CFC2=====',cfc2.size())

        cfc3 =self.squash( self.cfc3(x3), dim=-1)
        # print('CFC3=====',cfc3.size())

        x1= cfc1
        x2= cfc2
        x3= cfc3

        # x = torch.cat([cfc1, cfc2 ,cfc3], dim=1)


        # x = self.dropout(x, not(y is None))


     
        res1 = self.caps1(x1.view(x1.size(0), -1, self.fc_out_dim)).squeeze().transpose(0, 1)
        # print('RESSSSS11111',res1.size())


        res2 = self.caps2(x2.view(x2.size(0), -1, self.fc_out_dim)).squeeze().transpose(0, 1)
        # print('RESSSSS22222',res2.size())


        res3 = self.caps3(x3.view(x3.size(0), -1, self.fc_out_dim)).squeeze().transpose(0, 1)
        # print('RESSSSS33333',res3.size())



        # res=res1 + res2 + res3
        res= (res1 + res2 + res3)/3
        # print('RESSSSStotallllllll',res.size())


        classes = (res ** 2).sum(dim=-1) ** 0.5 #L2 norm of res (magnitude of output capsules)
        classes, reconstructions = self.decoder(res, y, classes)



        return classes, reconstructions


class CapsuleLoss(nn.Module):
    def __init__(self, hard, nc_recon):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)
        self.hard = hard
        self.nc_recon = nc_recon

    def forward(self, images, labels, classes, reconstructions):
        if self.hard:
            m_plus = 0.95
            m_minus = 0.05
            scaler = 0.8
        else:
            m_plus = 0.9
            m_minus = 0.1
            scaler = 0.5

        left = F.relu(m_plus - classes, inplace=True) ** 2
        right = F.relu(classes - m_minus, inplace=True) ** 2

        margin_loss = labels * left + scaler  * (1. - labels) * right
        margin_loss = margin_loss.sum()

        # print(f'img recon sizes {images.size()} {reconstructions.size()}')
        if self.nc_recon == 1:
            images = torch.sum(images, dim=1) / 3
        # print(f'img recon sizes {images.size()} {reconstructions.size()}')
        assert torch.numel(images) == torch.numel(reconstructions)
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)


if __name__ == "__main__":
    from torch.autograd import Variable
    from torch.optim import Adam
    from torchnet.engine import Engine
    from torchnet.logger import VisdomPlotLogger, VisdomLogger
    from torchvision.utils import make_grid
    from torchvision.datasets.mnist import MNIST, FashionMNIST
    from torchvision.datasets.svhn import SVHN
    from torchvision.datasets.cifar import CIFAR10

    from tqdm import tqdm
    import torchnet as tnt
    import argparse

    # parser = argparse.ArgumentParser(description="torchCapsNet.")
    # parser.add_argument('--dset', default='cifar')
    # parser.add_argument('--nc', default=10, type=int)
    # parser.add_argument('--w', default=32, type=int)
    # parser.add_argument('--dpath', default='')
    # parser.add_argument('--bsize', default=128, type=int)

    # parser.add_argument('--ne', default=100, type=int)
    # parser.add_argument('--niter', default=3, type=int)
    # parser.add_argument('--fck', default=1, type=int)
    # parser.add_argument('--fdim', default=8, type=int)
    # parser.add_argument('--ich', default=3, type=int)
    # parser.add_argument('--dec_type', default='DECONV')
    # parser.add_argument('--drp', default=0, type=float)
    # parser.add_argument('--res_folder', default='output-new')
    # parser.add_argument('--aug', default=1, type=int)
    # parser.add_argument('--nc_recon', default=1, type=int)
    # parser.add_argument('--hard', default=0, type=int)
    # parser.add_argument('--checkpoint', default='')
    # parser.add_argument('--test_only', default=0, type=int)
    # parser.add_argument('--lr', default=0.001, type=float)


    # args = parser.parse_args()

    parser = argparse.ArgumentParser(description="torchCapsNet.")
    parser.add_argument('--dset', default='mnist', required=True)
    parser.add_argument('--nc', default=10, type=int, required=True)
    parser.add_argument('--w', default=28, type=int, required=True)
    parser.add_argument('--dpath', default='')
    parser.add_argument('--bsize', default=128, type=int)

    parser.add_argument('--ne', default=100, type=int)
    parser.add_argument('--niter', default=3, type=int)
    parser.add_argument('--fck', default=1, type=int)
    parser.add_argument('--fdim', default=8, type=int, required=True)
    parser.add_argument('--ich', default=1, type=int, required=True)
    parser.add_argument('--dec_type', default='DECONV')
    parser.add_argument('--drp', default=0, type=float)
    parser.add_argument('--res_folder', default='output', required=True)
    parser.add_argument('--aug', default=1, type=int)
    parser.add_argument('--nc_recon', default=3, type=int, required=True)
    parser.add_argument('--hard', default=0, type=int, required=True)
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--test_only', default=0, type=int)



    args = parser.parse_args()

    resultsFolder = args.res_folder
    if not (os.path.exists(resultsFolder)):
        os.mkdir(resultsFolder)

    outputFolder = f'CFC-{args.dec_type}-{args.dset}-{args.fck}-{args.fdim}-{args.drp}'
    if args.hard == 1:
        outputFolder += '-H'

    expNoFile = f'{resultsFolder}/{outputFolder}/no'
    newExp = 1
    if os.path.exists(expNoFile):
        with open(expNoFile, 'r') as file:
            newExp = int(file.readline().replace('\n', '')) + 1
    else:
        if not os.path.exists(f'{resultsFolder}/{outputFolder}'):
            os.makedirs(f'{resultsFolder}/{outputFolder}')

    with open(expNoFile, 'w') as file:
        file.write(f'{newExp}\n')

    outputFolder = f'{resultsFolder}/{outputFolder}/{newExp}'
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)


    def processor(sample):

        data, labels, training = sample

        if (len(data.size()) == 3):
            data = data.unsqueeze(1)

        data = data.float() / 255.0
        if args.aug == 1:
            data = augmentation(data)

        if args.dset == 'aff_expanded':
            labels = labels.reshape(labels.size(0))

        labels = torch.LongTensor(labels)

        labels = torch.eye(args.nc).index_select(dim=0, index=labels)

        data = Variable(data).cuda()
        labels = Variable(labels).cuda()


        if training:
            classes, reconstructions = model(data, labels)
        else:
            classes, reconstructions = model(data)

        # print(f'!@@@@ reconstruction = {reconstructions.size()}')

        loss = capsule_loss(data, labels, classes, reconstructions)

        return loss, classes

   
    model = CapsuleNet(args.nc, args.niter, width=args.w, in_channels=args.ich, fc_kernel_size=args.fck,
                       fc_out_dim=args.fdim, nc_recon=args.nc_recon, decoder_type=args.dec_type, drp=args.drp)
    if args.hard == 1 or args.checkpoint != "":
        print(f'Loading Checkpoint: {args.checkpoint}')
        model.load_state_dict(torch.load(args.checkpoint))

    # model.load_state_dict(torch.load('epochs/epoch_327.pt'))
    model.cuda()

    # print("# parameters:", sum(param.numel() for param in model.parameters()))
    numparams = sum(param.numel() for param in model.parameters())
    print(f'### PARAMS = {numparams}\n')

    optimizer = Adam(model.parameters(), lr=args.lr)

    decayRate = 0.96
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)


    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
    train_error_logger = VisdomPlotLogger('line', opts={'title': 'Train Accuracy'})
    test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'})
    test_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Test Accuracy'})

    ground_truth_logger = VisdomLogger('image', opts={'title': 'Ground Truth'})
    reconstruction_logger = VisdomLogger('image', opts={'title': 'Reconstruction'})

    train_loss = []
    train_error = []
    test_loss = []
    test_accuracy = []

    if args.hard == 0:
        capsule_loss = CapsuleLoss(hard=False, nc_recon=args.nc_recon)

    else:
        capsule_loss = CapsuleLoss(hard=True, nc_recon=args.nc_recon)
    tr_ep_times = []
    tr_time_temp = 0
    tst_times = []
    final_acc = 0





    def reset_meters():
        meter_accuracy.reset()
        meter_loss.reset()



    def on_sample(state):
        state['sample'].append(state['train'])


    def on_forward(state):
        meter_accuracy.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        meter_loss.add(state['loss'].item())


    def on_start_epoch(state):
        reset_meters()
        state['iterator'] = tqdm(state['iterator'])
        global tr_time_temp
        tr_time_temp = time.time()


    def on_end_epoch(state):
        tr_ep_times.append(time.time() - tr_time_temp)
        print('[Epoch %d] Training Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

        train_loss.append(meter_loss.value()[0])
        train_loss_logger.log(state['epoch'], meter_loss.value()[0])
        train_error.append(meter_accuracy.value()[0])
        train_error_logger.log(state['epoch'], meter_accuracy.value()[0])

        reset_meters()

        temp = time.time()
        engine.test(processor, get_iterator(args.dset, args.bsize, False))
        tst_times.append(time.time() - temp)
        test_loss.append(meter_loss.value()[0])
        test_loss_logger.log(state['epoch'], meter_loss.value()[0])
        test_accuracy.append(meter_accuracy.value()[0])
        test_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
        my_lr_scheduler.step()

        print('[Epoch %d] Testing Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

        global final_acc
        final_acc = meter_accuracy.value()[0]

        if int(state['epoch']) == args.ne:
            torch.save(model.state_dict(), f'{outputFolder}/checkpoint.pt')
        # torch.save(model.state_dict(), f'{args.res_folder}/epoch_%d.pt' % state['epoch'])

        # Reconstruction visualization.

        #test_sample = next(iter(get_iterator(False)))

        #a = test_sample
        #if (len(a[0].size()) == 3):
        #    a[0] = a[0].unsqueeze(1)

        #ground_truth = (a[0].float() / 255.0)
        #_, reconstructions = model(Variable(ground_truth).cuda())
        #reconstruction = reconstructions.cpu().view_as(ground_truth).data

        #ground_truth_logger.log(
        #    make_grid(ground_truth, nrow=int(args.bsize ** 0.5), normalize=True, range=(0, 1)).numpy())
        #reconstruction_logger.log(
        #    make_grid(reconstruction, nrow=int(args.bsize ** 0.5), normalize=True, range=(0, 1)).numpy())


    # def on_start(state):
    #     state['epoch'] = 327
    #
    # engine.hooks['on_start'] = on_start
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    if args.test_only == 1:
        state = engine.test(processor, get_iterator(args.dset, args.bsize, False))
        print(f"Test Accuracy: {meter_accuracy.value()[0]}")
        quit()
    engine.train(processor, get_iterator(args.dset, args.bsize, True), maxepoch=args.ne, optimizer=optimizer)

    with open(f'{outputFolder}/res', 'w') as file:
        file.write(f'{final_acc}\n{np.mean(tr_ep_times)}\n{np.mean(tst_times)}\n{numparams}')

    with open(f'{outputFolder}/det', 'w') as outFile:
        outFile.write('#epoch,tr_acc,tst_acc,tr_loss,tst_loss\n')
        for i in range(len(train_loss)):
            outFile.write(f'{i + 1},{train_error[i]},{test_accuracy[i]},{train_loss[i]},{test_loss[i]}\n')

    if args.checkpoint == '':
        import os

        hard_cmd = f'python main.py --dset {args.dset} --w {args.w} --nc {args.nc} --ich {args.ich} --fck {args.fck} --fdim {args.fdim} --dec_type {args.dec_type} --res_folder {args.res_folder} --drp {args.drp} --aug {args.aug} --nc_recon {args.nc_recon} --hard 1 --checkpoint {outputFolder}/checkpoint.pt'
        with open('hardrun', 'w') as file:
            file.write(f'{hard_cmd}')