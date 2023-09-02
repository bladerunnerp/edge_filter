#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import PIL
import PIL.Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
from keras_cv import layers as kcv_layers
import cv2

tf.data.experimental.enable_debug_mode()

class GaussianKernel(tf.keras.initializers.Initializer):

    def _pre_computed(self):
        return np.array([[0.01582423, 0.01649751, 0.01672824, 0.01649751, 0.01582423],
       [0.01649751, 0.01719942, 0.01743997, 0.01719942, 0.01649751],
       [0.01672824, 0.01743997, 0.01768388, 0.01743997, 0.01672824],
       [0.01649751, 0.01719942, 0.01743997, 0.01719942, 0.01649751],
       [0.01582423, 0.01649751, 0.01672824, 0.01649751, 0.01582423]])
    
    def __init__(self, shape, sigma=1, in_channels=None):
        self.in_channels = in_channels
        self.sigma = sigma
        self.shape = shape
        
    def __call__(self, shape, dtype=None):
        kernel_weights = self._pre_computed()
        if self.in_channels is not None:
            kernel_weights = np.expand_dims(kernel_weights, axis=-1)
            kernel_weights = np.repeat(kernel_weights, self.in_channels, axis=-1)
            kernel_weights = np.expand_dims(kernel_weights, axis=-1) 
        return tf.constant(kernel_weights, dtype)

gaussian = layers.Conv2D(filters=1,kernel_size=(5,5), use_bias=False, padding='same', 
                        kernel_initializer=GaussianKernel(shape=(5,5), in_channels=3, sigma=3), name="gaussian")

class LaplacianKernel(tf.keras.initializers.Initializer):
    def __init__(self, in_channels=1):
        self.kernel = np.array([
                
                [-1,-1,-1,-1,-1],
                [-1,-1,-1,-1,-1],
                [-1,-1,24.3,-1,-1],
                [-1,-1,-1,-1,-1],
                [-1,-1,-1,-1,-1],
                
        ])
        self.in_channels = in_channels
        
    def __call__(self, shape, dtype=None):
        kernel_weights = self.kernel
        if self.in_channels is not None:
            kernel_weights = np.expand_dims(kernel_weights, axis=-1)
            kernel_weights = np.repeat(kernel_weights, self.in_channels, axis=-1)
            kernel_weights = np.expand_dims(kernel_weights, axis=-1) 
        return tf.constant(kernel_weights, dtype)

laplacian = layers.Conv2D(filters=1, kernel_size=5, use_bias=False, padding='same', 
                        kernel_initializer=LaplacianKernel(in_channels=1), name="laplacian")

class ThresholdLayer(layers.Layer):
    def __init__(self, thresh=None, **kwargs):
        super(ThresholdLayer, self).__init__(**kwargs)
        self.thresh = thresh
        
    def _threshold(self, a, thresh=None):
        mean = tf.numpy_function(np.mean, [a], a.dtype)
        if thresh is not None:
            mean = mean * thresh
        return tf.where(tf.less(a, tf.zeros_like(a) + mean), tf.zeros_like(a), tf.ones_like(a))
    
    def __call__(self, inputs):
        return self._threshold(inputs, self.thresh)

threshold = ThresholdLayer(thresh=0.02, name="threshold")

model = tf.keras.Sequential()
model.add(keras.Input(type_spec=tf.TensorSpec((1, 480, 640, 3))))
model.add(gaussian)
model.add(layers.Rescaling(scale=1./255))
model.add(laplacian)
model.add(threshold)
model.compile()

if __name__ == "__main__":
    vid = cv2.VideoCapture(0)
    frame_count = 0
    while(True):
        frame_count += 1
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)
        
        #the model wants a numpy array of images, we just wrap ours in one
        frame = np.array([frame])
        
        frame = model.predict(frame)
        
        #the model spits out a numpy array of images, we just strip it
        
        frame = frame[0,:,:,:]

        # Display the resulting frame
        cv2.imshow('frame', frame)
        
        key = cv2.waitKey(1) & 0xFF
        # if the q key was pressed, break from the loop
            
        if key == ord("q"):
            break
            
    vid.release()
    cv2.destroyAllWindows()
