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

# import deep neural network architecture claside_sizees
import Model_ResNet18
import Model_VGG16
import Model_VGG19
import Model_Inception_V3
import Model_Inception_V4

model_flag = Model_VGG16
mult = 150
side_size = 128

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

# Picaside_sizeo style image
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

# input videos

# Person Waving Hands content video
video_content_01 = cv2.VideoInput_Videoture("./Videos/video_content_01.mp4")

# Family Guys content video
video_content_01 = cv2.VideoInput_Videoture("./Videos/video_content_02.mp4")

# cartoon clip content video
video_content_03 = cv2.VideoInput_Videoture("./Videos/video_content_03.mp4")

Input_Video = video_content_01
img_con = None
img_list = []
img_con2 = None
i = 0

# Convert video into frames and preprocess
while(Input_Video.isOpened()):
  ret, im = Input_Video.read()
  if ret:
    frame = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame).resize((side_size,side_size))
    if img_con == None:
      img_con = img
    elif img_con2 == None:
      img_con2 = img
    if(len(img_list) < mult):
      img_list.append(img)
    if i > 87*30-15:
      break
    i += 1
  else:
    break

Input_Video.release()

# Check Image List
print(img_con.size)
img_sty = Image.open('picaside_sizeo.jpg').resize((side_size,side_size))
# display(img_list[-1].resize((480, 360)))
# display(img_sty.resize((480, 360)))

# Normalize frames
cnn_normalization_mean = jnp.array([0.485, 0.456, 0.406])
cnn_normalization_std = jnp.array([0.229, 0.224, 0.225])

image_content = []
print(len(img_list))
for i in img_list:
  image_content.append(jnp.array(i, dtype=jnp.float32) / 255.0)

# Add batch dimension
img_content = jnp.stack(image_content, axis=0)
print(img_content.shape)
image_style = jnp.array(img_sty, dtype=jnp.float32) / 255

#img_style = jnp.expand_dims(image_style, axis=0)
img_style = jnp.stack([image_style]*2)
print(img_style.shape)

# Model Class
class StyleTransfer:
  def __init__(self, input_content, input_style, content_layers, style_layers):
    # Original style and content input
    self.origin_content = input_content
    self.origin_style = input_style

    # Initialize Pretrained model
    self.vgg19 = fm.VGG16(output='activations', pretrained='imagenet', include_head=False)
    self.init_rngs = {'params': jax.random.PRNGKey(0)}
    self.vggparams = self.vgg19.init(self.init_rngs, img_content)
    self.fn_out = jit(self.vgg19.apply)

    # Get VGG activation of original style and content input
    self.activation_style_origin = self.fn_out(self.vggparams, self.origin_style, train=False)
    self.activation_content_origin = self.fn_out(self.vggparams, self.origin_content, train=False)

    # Initialize style layer and content layer
    self.layer_style = style_layers
    self.layer_content = content_layers

    #Initialize style and content weights
    self.style_weight = 1000000000
    self.content_weight = 1
    self.temporal_weight = 1000

    # Initialize Optimizer
    self.lr = 1e-3
    self.optimizer = optax.adam(learning_rate = self.lr)

    # Initialize generated image
    self.generate_img = self.origin_content.copy()
    self.opt_state = self.optimizer.init(self.generate_img)

  def reinit(self, input_content):
    # Original style and content input
    self.origin_content = input_content

    # Initialize Pretrained model
    self.vgg19 = fm.VGG19(output='activations', pretrained='imagenet', include_head=False)
    self.init_rngs = {'params': jax.random.PRNGKey(0)}#, 'dropout': jax.random.PRNGKey(1)}
    self.vggparams = self.vgg19.init(self.init_rngs, img_content)
    self.fn_out = jit(self.vgg19.apply)

    # Get VGG activation of original style and content input
    self.activation_style_origin = self.fn_out(self.vggparams, self.origin_style, train=False)
    self.activation_content_origin = self.fn_out(self.vggparams, self.origin_content, train=False)

    #Initialize style and content weights
    self.style_weight = 1000000000
    self.content_weight = 1
    self.temporal_weight = 1000

    # Initialize Optimizer
    self.lr = 1e-3
    self.optimizer = optax.adam(learning_rate = self.lr)

    # Initialize generated image
    self.generate_img = self.origin_content.copy()
    self.opt_state = self.optimizer.init(self.generate_img)

  @partial(jit, static_argnums=(0,))
  # function to compute Gram Matrix
  def gram_matrix(self, input):
    # transfer tensor dimensions: (#N, #C, #H, #W) => (#N, #H, #W, #C)
    input = jnp.transpose(input, axes=(0, 3, 1, 2))
    # batch, channel, height, weight of the input images
    batch, channel, height, weight = input.shape
    # reshape input features
    features = input.reshape(batch * channel, height * weight)
    # Gram Matrix = V @ V.T
    G = features @ features.T
    return G / (batch * channel * height * weight)

  @partial(jit, static_argnums=(0,))
  # function to compute content_loss
  def content_loside_size(self, input_content, img_generated):
    # content_loss = (content-Generated)^2
    return jnp.mean((input_content.flatten()-img_generated.flatten()) ** 2)

  @partial(jit, static_argnums=(0,))
  # function to compute style_loss
  def temporal_loside_size(self, input_content, img_generated):
    # content_loss = (style-Generated)^2
    return jnp.mean((img_generated[0].flatten()-img_generated[1].flatten()) ** 2)

  @partial(jit, static_argnums=(0,))
  def style_loside_size(self, input_style, img_generated):
    return jnp.mean((input_style - img_generated) ** 2)

  @partial(jit, static_argnums=(0,))
  def loside_size(self, img_generated):
    out_generated = self.fn_out(self.vggparams, img_generated, train=False)
    style_score = 0
    content_score = 0
    temporal_score = 0

    for cont_layer in self.layer_content:
      content_score += self.content_loside_size(self.activation_content_origin[cont_layer], out_generated[cont_layer])
      temporal_score += self.temporal_loside_size(self.activation_content_origin[cont_layer], out_generated[cont_layer])

    for sty_layer in self.layer_style:
      gram_sty = self.gram_matrix(self.activation_style_origin[sty_layer])
      gram_gen = self.gram_matrix(out_generated[sty_layer])
      style_score += self.style_loside_size(gram_sty, gram_gen)

    loside_size = self.style_weight * style_score + self.content_weight * content_score + self.temporal_weight * temporal_score
    return loside_size

  @partial(jit, static_argnums=(0,))
  def step(self, optimizer_state, img_generated):
    grads = grad(self.loside_size)(img_generated)
    updates, opt_state = self.optimizer.update(grads, optimizer_state, img_generated)
    return optax.apply_updates(img_generated, updates), opt_state

  def train(self, imgs, iter = 5000):
    retArray = []

    for i in trange(imgs.shape[0]-1):
      self.reinit(imgs[i:i+2,:,:,:])
      for iter in range(5000):
        self.generate_img, self.opt_state = self.step(self.opt_state, self.generate_img)
        self.generate_img = jnp.clip(self.generate_img, 0, 1)
      retArray.append(jnp.clip(self.generate_img, 0, 1))
    return retArray

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

sty_trans = StyleTransfer(img_content[:2], img_style, content_layers_default, style_layers_default)
output = sty_trans.train(img_content)

print(output[0].shape)
videodims = (output[0].shape[1], output[0].shape[2])
fourcc = cv2.VideoWriter_fourcc("F","M","P","4")
video = cv2.VideoWriter("./Videos/Output_Video.mp4",fourcc, 30,videodims)
img = Image.new('RGB', videodims, color = 'darkred')

ii = None
iii = None

# Add to each frame of video
for i in trange(0,len(output)):
    imtemp = output[i][0,:,:,:].copy()
    if i > 0:
      imtemp += output[i-1][1,:,:,:].copy()
      imtemp /= 2
    if i == 10:
      ii = imtemp.copy()
    if i == 11:
      iii = imtemp.copy()
    video.write(cv2.cvtColor(np.uint8(np.array(imtemp)*255), cv2.COLOR_RGB2BGR))
video.release()

x = np.abs(np.asarray(ii-iii))*25*100
print(type(x))

# ouput shotcut image
cv2.imwrite('Output_Video.jpg', x)
plt.imshow(output[1][0])