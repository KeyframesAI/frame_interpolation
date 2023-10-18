import gradio as gr
from interpolate import run_model, save_outputs
import time
from random import random 
import os
from PIL import Image
import json


def run(frame1, frame2):
    interp, start, end = run_model('prod', frame1, frame2)
    
    folder = 'output/'+str(time.time()).replace('.', '_')+'-'+str(int(random()*100000))+'/'
    save_outputs(interp, start, end, folder)
    
    filenames = os.listdir(folder)
    imgs = []
    for f in filenames:
        if f != 'a.png' and f != 'c.png':
            imgs.append(Image.open(folder + f))
    return imgs

gr.Interface(fn=run,
    inputs=[gr.Image(type="pil"), gr.Image(type="pil")],
    outputs=gr.Gallery(columns=10),
    examples=[
        ['test/frame_000000000.jpg', 'test/frame_000000001.jpg'],
        ['test/frame_000000002.jpg', 'test/frame_000000003.jpg'],
        ['test/frame_000000004.jpg', 'test/frame_000000005.jpg'],
        ['test/frame_000000006.jpg', 'test/frame_000000007.jpg'],
        ['test/frame_000000008.jpg', 'test/frame_000000009.jpg'],
        ['test/frame_000000010.jpg', 'test/frame_000000011.jpg'],
        ['test/frame_000000012.jpg', 'test/frame_000000013.jpg'],
        ['test/frame_000000014.jpg', 'test/frame_000000015.jpg'],
    ]).launch(share=True)
