# Frame Interpolation

This implementation builds on top of existing code: [Self-Attention-GAN](https://github.com/heykeetae/Self-Attention-GAN) and [frame_interpolation_GAN](https://github.com/tnquang1416/frame_interpolation_GAN)

It uses latent space interpolation for a GAN trained on video frames

## Results

![results](https://github.com/KeyframesAI/frame_interpolation/blob/main/output/interp.gif)

## How to run

### Data

Trained on a dataset of video frames available [here](https://drive.google.com/u/0/uc?id=18FbMur-goI6ZnWh-fP0SNuq8DVCnMyKn)

To extract frames from your own videos:

1. Save the videos in `data/temp/`

2. Run:
```
cd data/
python video2frames.py
```

### Training

```
python main.py --log_step 100 --sample_step 1000 --total_step 1000000 --batch_size 16 --imsize 128 --dataset frames --adv_loss hinge --version fr16
```

To continue training an existing model (eg. `50184_G.pth`):

```
python main.py --log_step 100 --sample_step 1000 --total_step 10000000 --batch_size 16 --imsize 128 --dataset frames --adv_loss hinge --version fr16 --pretrained_model 50184
```

### Testing

The input frames are in `test/`

```
python test.py
```
