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

# deep neural network class
class StyleTransfer:
  def __init__(self, content_image, style_image, content_layers, style_layers):

    # Initialize original content and style image inputs
    self.origin_content = content_image
    self.origin_style = style_image

    # Initialize Deep Neural Network
    self.resnet18 = fm.ResNet18(output="activations", pretrained="imagenet")
    self.init_rngs = {"params": jax.random.PRNGKey(0)}
    self.resparams = self.resnet18.init(self.init_rngs, content_image)
    self.fn_out = self.resnet18.apply

    # Initialize activation of original content and style image inputs
    self.activation_style_origin = self.fn_out(self.resparams, self.origin_style, train=False)
    self.activation_content_origin = self.fn_out(self.resparams, self.origin_content, train=False)

    # Initialize style layers and content layers
    self.layer_style = style_layers
    self.layer_content = content_layers

    # Initialize style and content weights
    # Tune these parameters to balance output
    self.style_weight=100000
    self.content_weight=1

    # Setup Optimizer, using adam algorithm
    self.lr = 1e-3 # 1e-4, 1e-5
    self.optimizer = optax.adam(learning_rate = self.lr)

    # Initialize Output Images
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
  def content_loss(self, content_image, img_generated):
    # content_loss = (content-Generated)^2
    return jnp.mean((content_image.flatten() - img_generated.flatten())**2)

  @partial(jit, static_argnums=(0,))
  # function to compute style_loss
  def style_loss(self, style_image, img_generated):
    # content_loss = (style-Generated)^2
    return jnp.mean((style_image - img_generated) ** 2)

  @partial(jit, static_argnums=(0,))
  # function to loss function
  def loss(self, img_generated):
    out_generated = self.fn_out(self.resparams, img_generated, train=False)
    style_score = 0
    content_score = 0

    # compute content score
    for cont_layer in self.layer_content:
      content_score += self.content_loss(self.activation_content_origin[cont_layer], out_generated[cont_layer])

    # compute style score
    for sty_layer in self.layer_style:
      gram_sty = self.gram_matrix(self.activation_style_origin[sty_layer])
      gram_gen = self.gram_matrix(out_generated[sty_layer])
      style_score += self.style_loss(gram_sty, gram_gen)

    # compute overall loss
    # loss = style_weight * style_score + content_weight * content_score
    loss = self.style_weight * style_score + self.content_weight * content_score
    return loss

  # forward helper function
  @partial(jit, static_argnums=(0,))
  def step(self, optimizer_state, img_generated):
    grads = grad(self.loss)(img_generated)
    # initialize optimizer
    updates, opt_state = self.optimizer.update(grads, optimizer_state, img_generated)
    return optax.apply_updates(img_generated, updates), opt_state

  # helper function to train network
  def train(self, iter = 2000):
    for iter in trange(iter):
      self.generate_img, self.opt_state = self.step(self.opt_state, self.generate_img)
      self.generate_img = jnp.clip(self.generate_img, 0, 1)

    return self.generate_img

