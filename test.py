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
from interpolate import run_model


def save_outputs(interp, start, end, output_path='output/'):

    save_image(start, output_path+'a.png')
    save_image(end, output_path+'c.png')

    for i in range(len(interp)): 
        save_image(interp[i], output_path+'b'+str(i)+'.png')

    print('generated images saved in: '+output_path)



im1 = Image.open(os.path.join('test', 'frame_000000008.jpg'))
im2 = Image.open(os.path.join('test', 'frame_000000009.jpg'))

interp, start, end = run_model('prod', im1, im2)
save_outputs(interp, start, end)
