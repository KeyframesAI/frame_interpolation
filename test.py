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


model_save_path = 'models'
input_path = 'test'
output_path = 'output/'
batch_size = 16
nimgs = 8
interp_pairs = True
pretrained_model = '436896' # '70848'
use_prod_model = True


def build_model():
    G = Generator(batch_size,128, 100, 64).cuda()
    #D = DiscriminatorPix2Pix(9, d_conv_dim, 1, True).cuda()
    D = Discriminator(batch_size,128, 64).cuda()

    # print networks
    print(G)
    print(D)
    return G, D

def load_pretrained_model(G, D):
    if use_prod_model:
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


G, D = build_model()
G, D = load_pretrained_model(G, D)

#data_loader = torch.utils.data.DataLoader(dataset=frame_dataset.FrameDataset(128), batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
data_loader = torch.utils.data.DataLoader(dataset=frame_dataset.FrameDataset(128, input_path), batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
data_iter = iter(data_loader)



imgs, _ = next(data_iter)
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

if interp_pairs:
    save_image(getAlternate(inputs)[:nimgs], output_path+'a.png')
    save_image(getAlternate(inputs, False)[:nimgs], output_path+'c.png')
else:
    save_image(inputs[:nimgs], output_path+'a.png')

for i in range(len(factors)): 
    save_image(decoded[i][:nimgs], output_path+'b'+str(i)+'.png')

print('generated images saved in: '+output_path)
