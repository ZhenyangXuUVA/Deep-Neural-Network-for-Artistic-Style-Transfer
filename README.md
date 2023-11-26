## Deep-Neural-Network-for-Artistic-Style-Transfer
Deep Neural Network for Artistic Style Transfer

## Introduction
In this work, we researched the deep neural network architectures for artistic style transfer. A loss function with multiple regularization terms is proposed to construct the neural network. Pre-trained model with different architectures are used during our network training phase. In order to better understaing and optimize the deep neural network architecture, we experimented and tested the effect caused be picking different GD algorithms, ratio parameters and DNN Architectures to final output images. Overall, the results generated through Adam method with VGG19 architecture and ratio parameter set between 100-1000 looks reasonable and cost affordable computation resources within this project. For video style transfer, we developed a process to style transfer small videos with low resolution. A new regularization term is proposed to keep the continuity of consecutive frames within the video. The output video shows reasonable result.

## Requirement
Packages required for this program is lised below: 
- [OpenCV](https://github.com/opencv/opencv): A package used for image and video processing
- [Optax](https://github.com/deepmind/optax): A gradient processing and optimization library
- [PIL](https://github.com/python-pillow/Pillow): A package for image processing
- [Google JAX](https://github.com/google/jax): A package for image formating
- [Flax](https://github.com/google/flax): A neural network library

## Code files:
- Process_Image.py: code frame to process images
- Process_Video.py: code frame to process videos
- Model_ResNet18.py: class code to use ResNet18 deep neural network architechture
- Model_VGG16.py: class code to use VGG16 deep neural network architechture
- Model_VGG19.py: class code to use VGG19 deep neural network architechture
- Model_Inception_V3.py: class code to use InceptionV3 deep neural network architechture
- Model_Inception_V4.py: class code to use InceptionV4 deep neural network architechture

## Data files:
- Images: test content and style input images
- Videos: test content videos
- Output_Videos: styled ouput videos
  - DizzyEffect0: content video
  - DizzyEffect1: output video (Picasso Style)
  - DizzyEffect1: output video (Starry Night Style)
  - DizzyEffect1: output video (The Screamer Style)

## Image Results:
<img src="https://github.com/ZhenyangXuUVA/Deep-Neural-Networks-for-Artistic-Style-Transfer/blob/main/Readme/Figure01.png" width="1000" height="516">






