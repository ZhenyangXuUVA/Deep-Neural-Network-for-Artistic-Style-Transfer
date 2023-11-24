# basic libraries
import cv2
import torch
import optax
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import trange
from functools import partial

# libraries for image jax format
import jax
import jax.numpy as jnp
import flaxmodels as fm
from jax import jit, random, grad
from jax.example_libraries import optimizers

# import deep neural network architecture classes
import Model_ResNet18
import Model_VGG16
import Model_VGG19
import Model_Inception_V3
import Model_Inception_V4

model_flag = Model_ResNet18

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# input all content & style images

# ballet dancer content image
image_content_01 = Image.open('./Images/image_content_01.jpg').resize((1024,1024))
# display(image_content_01.resize((512, 512)))

# Mona Lisa content image
image_content_02 = Image.open('./Images/image_content_02.jpg').resize((1024,1024))
# display(image_content_02.resize((512, 512)))

# Lamborghini content image
image_content_03 = Image.open('./Images/image_content_03.jpg').resize((1024,1024))
# display(image_content_03.resize((512, 512)))

# Picasso style image
image_style_01 = Image.open('./Images/image_style_01.jpg').resize((1024,1024))
# display(image_style_01.resize((512, 512)))

# Starry Night style image
image_style_02 = Image.open('./Images/image_style_02.jpg').resize((1024,1024))
# display(image_style_02.resize((512, 512)))

# Maple Forest style image
image_style_03 = Image.open('./Images/image_style_03.jpg').resize((1024,1024))
# display(image_style_03.resize((512, 512)))

# normalize content image pixel values into [0.0, 1.0]
image_content = jnp.array(image_content_01, dtype=jnp.float32) / 255.0
img_content = jnp.expand_dims(image_content, axis=0)

# normalize style image pixel values into [0.0, 1.0]
image_style = jnp.array(image_style_01, dtype=jnp.float32) / 255.0
img_style = jnp.expand_dims(image_style, axis=0)

# initialize model class
Model = StyleTransfer()

# configuration for Model_ResNet18
if model_flag == Model_ResNet18:
  content_layers_default = ['block3_1']
  style_layers_default = ['conv1', 'block1_1', 'block2_1', 'block3_1', 'block4_1']
  sty_trans = Model(img_content, img_style, content_layers_default, style_layers_default)

# configuration for Model_VGG16
if model_flag == Model_VGG16:
  content_layers_default = ['conv4_2']
  style_layers_default = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
  sty_trans = Model(img_content, img_style, content_layers_default, style_layers_default)

# configuration for Model_VGG19
if model_flag == Model_VGG19:
  content_layers_default = ['conv4_2']
  style_layers_default = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
  sty_trans = Model(img_content, img_style, content_layers_default, style_layers_default)

# configuration for Model_Inception_V3
if model_flag == Model_Inception_V3:
  content_layers_default = ['conv5_1']
  style_layers_default = ['conv2_1', 'conv2_1', 'conv4_1', 'conv4_1']
  sty_trans = Model(img_content, img_style, content_layers_default, style_layers_default)

# configuration for Model_Inception_V4
if model_flag == Model_Inception_V4:
  content_layers_default = ['conv5_1']
  style_layers_default = ['conv2_1', 'conv2_1', 'conv4_1', 'conv4_1']
  sty_trans = Model(img_content, img_style, content_layers_default, style_layers_default)

# Start Training 
output = sty_trans.train()

# output resulting image
img_output = np.array(output[0]*255).astype('uint8')
img_out = Image.fromarray(img_output)
# display(img_out)
Image.Image.save(img_out, fp='ResNet18.jpg')