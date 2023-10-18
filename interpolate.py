import os
import time
import torch
import datetime
from skimage.metrics import structural_similarity
from PIL import Image
import json
import numpy as np

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as T

from sagan_models import Generator, Discriminator
from utils import *
import frame_dataset





def build_model(batch_size):
    G = Generator(batch_size,128, 100, 64).cuda()
    #D = DiscriminatorPix2Pix(9, d_conv_dim, 1, True).cuda()
    D = Discriminator(batch_size,128, 64).cuda()

    # print networks
    #print(G)
    #print(D)
    return G, D

def load_pretrained_model(G, D, model_save_path, pretrained_model):
    if pretrained_model == 'prod':
        G.load_state_dict(torch.load(os.path.join(
            model_save_path, 'generator.pth')))
        D.load_state_dict(torch.load(os.path.join(
            model_save_path, 'discriminator.pth')))
        print('loaded prod models')
        return G, D
        
    G.load_state_dict(torch.load(os.path.join(
        model_save_path, '{}_G.pth'.format(pretrained_model))))
    D.load_state_dict(torch.load(os.path.join(
        model_save_path, '{}_D.pth'.format(pretrained_model))))
    print('loaded trained models (step: {})..!'.format(pretrained_model))
    return G, D
    
def getAlternate(s, getEvens = True):
    start = 0
    if not getEvens:
        start = 1
    return [s[i] for i in range(start, len(s), 2)]



def run_model(pretrained_model='prod', img1=None, img2=None):
    model_save_path = 'models'
    input_path = 'test'
    batch_size = 16
    nimgs = 8
    interp_pairs = True
    
    replace_imgs = img1 is not None and img2 is not None

    G, D = build_model(batch_size)
    G, D = load_pretrained_model(G, D, model_save_path, pretrained_model)

    if replace_imgs:
        #input_path = 'data/frames'
        #data_loader = torch.utils.data.DataLoader(dataset=frame_dataset.FrameDataset(128, input_path), batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
        data_loader = torch.utils.data.DataLoader(dataset=frame_dataset.FrameDataset(128, input_path), batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    else:
        data_loader = torch.utils.data.DataLoader(dataset=frame_dataset.FrameDataset(128, input_path), batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    data_iter = iter(data_loader)

    imgs, _ = next(data_iter)
    print(len(imgs))
    if replace_imgs:
        transTensor = T.ToTensor()
        imgs[0] = transTensor(img1)
        imgs[1] = transTensor(img2)
        nimgs = 1
    inputs = getFrames(imgs)

    latent = G.encode(inputs)

    factors = np.arange(0, 1, 0.1)
    decoded = []
    for factor in factors:
        ids = range(0, batch_size-1, 1)
        if interp_pairs:
            ids = range(0, batch_size, 2)

        interp_latent = latent.detach().clone()
        for i in ids:
            interp = latent[i] * (1 - factor) + latent[i+1] * factor
            interp_latent[i] = interp
            if interp_pairs:
                interp_latent[i+1] = interp

        dec = G.decode(interp_latent).to('cpu')
        if interp_pairs:
            dec = getAlternate(dec)
        decoded.append(dec)
    
    for i in range(len(factors)): 
        decoded[i] = decoded[i][:nimgs]
    
    return decoded, getAlternate(inputs)[:nimgs], getAlternate(inputs, False)[:nimgs]
    

def save_outputs(interp, start, end, output_path='output/'):
    #output_path = 'output/'
    if (not os.path.isdir(output_path)):
        os.mkdir(output_path)

    save_image(start, output_path+'a.png')
    save_image(end, output_path+'c.png')

    for i in range(len(interp)): 
        save_image(interp[i], output_path+'b'+str(i)+'.png')

    print('generated images saved in: '+output_path)

