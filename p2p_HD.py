from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
import scipy.ndimage as sn 
import scipy.io as sio

import warnings
from functools import partial
from nets import resnet_utils
from options.train_options import TrainOptions
from models import *
from vgg import * 
import ops

slim = tf.contrib.slim
resnet_arg_scope = resnet_utils.resnet_arg_scope

a = TrainOptions().parse()

EPS = 1e-12

NUM_SAVE_IMAGE = 100


Examples = collections.namedtuple("Examples", "filenames, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, beta_list, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_loss_fm, gen_grads_and_vars, train")
Tower = collections.namedtuple("Tower", "inputs, targets, outputs, predict_real, predict_fake, discrim_loss, gen_loss,gen_loss_GAN, gen_loss_L1, gen_loss_fm")
seed = random.randint(0, 2**31 - 1) 

##################### Data #####################################################

def transform(image, flip=a.flip, monochrome=a.monochrome, random_crop=a.random_crop):
    """ Transform image to augment data.
    Including:
        flip: flip image horizontally
        monochrome: rgb to grayscale
        center_crop: center crop image to make sure weight == height
        random_crop: resize image to a larger scale_size and randomly crop it to target a.target_size.
        resize: resize image to [a.scale_size, a.scale_size]        
    """
    # make a copy of image, otherwise get wrong results for unkwon reason
    r = image
    height = r.get_shape()[0] # h, w, c
    width = r.get_shape()[1]
    if flip:
        r = tf.image.random_flip_left_right(r, seed=seed)
    if monochrome:
        r = tf.image.rgb_to_grayscale(r)
    if not height == width:
        # center crop to correct ratio
        size = min(height, width)
        oh = (height - size) // 2
        ow = (width - size) // 2
        r = tf.image.crop_to_bounding_box(image=r, offset_height=oh, offset_width=ow, target_height=size, target_width=size)
    if  random_crop: 
        # resize to a.scale_size and then randomly crop to a.target_size
        r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)
        if not a.target_size == a.scale_size:
            offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - a.target_size + 1, seed=seed)), dtype=tf.int32)
            if a.scale_size > a.target_size:
                r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], a.target_size, a.target_size)
            elif a.scale_size < a.target_size:
                raise Exception("scale size cannot be less than crop size") 
    else:
        # resize to a.target_size
        r = tf.image.resize_images(r, [a.target_size, a.target_size], method=tf.image.ResizeMethod.AREA)

    return r 

def parse_function_test(example_proto):        
    features = {
            'filename': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'photo': tf.FixedLenFeature([], tf.string),
            # 'mask': tf.FixedLenFeature([], tf.string),
            'edge': tf.FixedLenFeature([], tf.string),
            'df': tf.FixedLenFeature([], tf.string)
            }        
    
    parsed_features = tf.parse_single_example(example_proto, features=features) 
    
    filenames = tf.decode_raw(parsed_features['filename'], tf.uint8)
    photo = tf.decode_raw(parsed_features['photo'], tf.uint8)
    photo = tf.reshape(photo, [218, 178, 3])  
    edge = tf.decode_raw(parsed_features['edge'], tf.float32) 
    edge = tf.reshape(edge, [218, 178, 1])
    df = tf.decode_raw(parsed_features['df'], tf.float64) 
    df = tf.reshape(df, [218, 178, 1])   
    
    photo = tf.image.convert_image_dtype(photo, dtype=tf.float64)
    photo = photo * 2. -1.     
    
    edge = (edge) * 2. - 1.
    df = df/tf.reduce_max(df)
    df = (df) * 2. - 1.
    
    height = parsed_features['height']
    width = parsed_features['width']
    print(height, width)
  
    edge = transform(tf.image.grayscale_to_rgb(edge))
    df = transform(tf.image.grayscale_to_rgb(df))
    photo = transform(photo)     
    
    if a.input_type == "df":
        condition = df
    elif a.input_type == "edge": 
        condition = edge

    return photo, condition, filenames 

def parse_function_test_hd(example_proto):        
    features = {
            'filename': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'photo': tf.FixedLenFeature([], tf.string),
            # 'mask': tf.FixedLenFeature([], tf.string),
            'edge': tf.FixedLenFeature([], tf.string),
            'df': tf.FixedLenFeature([], tf.string)
            }        
    
    parsed_features = tf.parse_single_example(example_proto, features=features) 
    
    filenames = tf.decode_raw(parsed_features['filename'], tf.uint8)
    photo = tf.decode_raw(parsed_features['photo'], tf.uint8)
    photo = tf.reshape(photo, [218, 178, 3])  
    edge = tf.decode_raw(parsed_features['edge'], tf.float32) 
    edge = tf.reshape(edge, [218, 178, 1])
    df = tf.decode_raw(parsed_features['df'], tf.float64) 
    df = tf.reshape(df, [218, 178, 1])   
    
    photo = tf.image.convert_image_dtype(photo, dtype=tf.float64)
    photo = photo * 2. -1.     
    
    edge = (edge) * 2. - 1.
    df = df/tf.reduce_max(df)
    df = (df) * 2. - 1.
    
    height = parsed_features['height']
    width = parsed_features['width']
    print(height, width)
  
    edge = transform(tf.image.grayscale_to_rgb(edge))
    df = transform(tf.image.grayscale_to_rgb(df))
    photo = transform(photo)     
    
    if a.input_type == "df":
        condition = df
    elif a.input_type == "edge": 
        condition = edge

    return photo, condition, filenames 

def parse_function(example_proto):
    '''
    Mask is not used, for future applications.    
    '''            
    features = {
            'filename': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'photo': tf.FixedLenFeature([], tf.string),
            'mask': tf.FixedLenFeature([], tf.string),
            'edge': tf.FixedLenFeature([], tf.string),
            'df': tf.FixedLenFeature([], tf.string)
            }        
    
    parsed_features = tf.parse_single_example(example_proto, features=features) 
    
    
    filenames = tf.decode_raw(parsed_features['filename'], tf.uint8)
    photo = tf.decode_raw(parsed_features['photo'], tf.uint8)
    photo = tf.reshape(photo, [218, 178, 3])  
    # mask = tf.decode_raw(parsed_features['mask'], tf.uint8)
    # mask = tf.reshape(mask, [218, 178, 1])
    edge = tf.decode_raw(parsed_features['edge'], tf.float32) 
    edge = tf.reshape(edge, [218, 178, 1])
    df = tf.decode_raw(parsed_features['df'], tf.float64) 
    df = tf.reshape(df, [218, 178, 1])   
    
    photo = tf.image.convert_image_dtype(photo, dtype=tf.float64)
    photo = photo * 2. -1.    
    #mask = tf.image.convert_image_dtype(mask, dtype=tf.float64)
    #mask = mask * 2. -1.  
    
    
    edge = (edge) * 2. - 1.
    df = df/tf.reduce_max(df)
    df = (df) * 2. - 1.
    
    height = parsed_features['height']
    width = parsed_features['width']
    print(height, width)

    seed = random.randint(0, 2**31 - 1)

    edge = transform(tf.image.grayscale_to_rgb(edge))
    df = transform(tf.image.grayscale_to_rgb(df))
    photo = transform(photo)   
    # mask = transform(mask)   
        
    if a.input_type == "df":
        condition = df
    elif a.input_type == "edge": 
        condition = edge

    return photo, condition, filenames

def parse_function_hd(example_proto):
    '''
     
    '''            
    features = {
            'filename': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'photo': tf.FixedLenFeature([], tf.string),
            'hed': tf.FixedLenFeature([], tf.string),
            'edge': tf.FixedLenFeature([], tf.string),
            'df': tf.FixedLenFeature([], tf.string)
            }        
    
    parsed_features = tf.parse_single_example(example_proto, features=features) 
    
    
    filenames = tf.decode_raw(parsed_features['filename'], tf.uint8)
    photo = tf.decode_raw(parsed_features['photo'], tf.uint8)
    photo = tf.reshape(photo, [512, 512, 3])  
    photo = tf.image.convert_image_dtype(photo, dtype=tf.float32)
    photo = photo * 2. -1.
    height = parsed_features['height']
    width = parsed_features['width']
    depth = parsed_features['depth']
    print(height, width, depth)

    photo = transform(photo)   

    if a.input_type == "df":
        df = tf.decode_raw(parsed_features['df'], tf.float32) 
        df = tf.reshape(df, [512, 512, 1])   
        #df = df/tf.reduce_max(df) # normalize the distance fields, by the max value, to fit grayscale
        df = df / a.df_norm_value # normalize the distance fields, by a given value, to fit grayscale
        df = (df) * 2. - 1.    
        df = transform(tf.image.grayscale_to_rgb(df))
        condition = df

    elif a.input_type == "edge": 
        edge = tf.decode_raw(parsed_features['edge'], tf.uint8) 
        edge = tf.reshape(edge, [512, 512, 1])
        edge = tf.image.convert_image_dtype(edge, dtype=tf.float32)
        edge = (edge) * 2. - 1.
        edge = transform(tf.image.grayscale_to_rgb(edge))
        condition = edge

    elif a.input_type == "hed": 
        hed = tf.decode_raw(parsed_features['hed'], tf.float32) 
        hed = tf.reshape(hed, [512, 512, 1])
        hed = (hed) * 2. - 1.
        hed = transform(tf.image.grayscale_to_rgb(hed))
        condition = hed
        
    elif a.input_type == "vg": 
        hed = tf.decode_raw(parsed_features['hed'], tf.float32) 
        hed = tf.reshape(hed, [512, 512, 1])
        #hed = (hed) * 2. - 1.
        #hed = transform(tf.image.grayscale_to_rgb(hed))
        edge = tf.decode_raw(parsed_features['edge'], tf.uint8) 
        edge = tf.reshape(edge, [512, 512, 1])
        edge = tf.image.convert_image_dtype(edge, dtype=tf.float32)
        edge = 1. - edge
        #edge = transform(tf.image.grayscale_to_rgb(edge))

        vg = tf.multiply(hed, edge)
        vg = tf.less(vg, tf.ones(tf.shape(vg)) * 1e-10)
        vg = ops.distance_transform(vg)
        vg = tf.reshape(vg, [512, 512, 1])
        #vg = vg / a.df_norm_value
        vg = vg / tf.reduce_max(vg)
        #vg = 2. - vg * 2.
        vg = transform(tf.image.grayscale_to_rgb(vg))

        print(vg.get_shape())

        condition = vg
    return photo, condition, filenames

def read_tfrecord():
    tfrecord_fn = glob.glob(os.path.join(a.input_dir, "*.tfrecords"))
    dataset = tf.data.TFRecordDataset(tfrecord_fn)
    if a.hd:
        if a.mode=='train':
            dataset = dataset.map(parse_function_hd)  # Parse the record into tensors. 
        else:
            dataset = dataset.map(parse_function_test_hd)  # Parse the record into tensors. If test, mask is not included in tfrecord file.
    else:
        if a.mode=='train':
            dataset = dataset.map(parse_function)  # Parse the record into tensors. 
        else:
            dataset = dataset.map(parse_function_test)  # Parse the record into tensors. If test, mask is not included in tfrecord file.
        
    dataset = dataset.repeat()  # Repeat the input indefinitely.
    # dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(a.batch_size)
    iterator = dataset.make_one_shot_iterator()
    photo, condition, filename = iterator.get_next()

    photo.set_shape([a.batch_size, a.target_size, a.target_size, 3])
    condition.set_shape([a.batch_size, a.target_size, a.target_size, 3])
    
    
    # print(condition.get_shape())
    steps_per_epoch = int(math.ceil(a.num_examples / a.batch_size))
    
    # show read results for code test
    #sess = tf.Session()
    #photo1, mat1 = sess.run(iterator.get_next())
    #Image.fromarray(photo1[0,:,:,:], "RGB").save('/data4T1/liyh/data/CelebA/tfrecord/a1.jpg')
    #sio.savemat(os.path.join('/data4T1/liyh/data/CelebA/tfrecord', "b2.mat"), {"predict": photo1[0,:,:,:]})
    #sio.savemat(os.path.join('/data4T1/liyh/data/CelebA/tfrecord', "a2.mat"), {"predict": mat1[0,:,:,:]})
    
    return Examples(
        filenames=filename,
        inputs=condition,
        targets=photo,
        count=len(tfrecord_fn),
        steps_per_epoch=steps_per_epoch
    ), iterator

##################### Generators #################################################

def create_generator_resgan(generator_inputs, generator_outputs_channels, gpu_idx=0):
    """
    gpu_idx: gpu index, use gpu with index of gpu_idx and gpu_idx+1

    """
    with tf.device("/gpu:%d" % (gpu_idx)):
        with tf.variable_scope("encoder"): 
            net = ops.conv(generator_inputs, channels=a.ngf, kernel=7, stride=1, pad=3, use_bias=True, sn=a.sn, scope='encoder_0')
            net = tf.contrib.layers.instance_norm(net)
            net = tf.nn.relu(net)
            print(net.get_shape())

            net = ops.conv(net, channels=a.ngf*2, kernel=4, stride=2, pad=1, use_bias=True, sn=a.sn, scope='encoder_1')
            net = tf.contrib.layers.instance_norm(net)
            net = tf.nn.relu(net)
            print(net.get_shape())

            net = ops.conv(net, channels=a.ngf*4, kernel=4, stride=2, pad=1, use_bias=True, sn=a.sn, scope='encoder_2')
            net = tf.contrib.layers.instance_norm(net)
            net = tf.nn.relu(net)
            print(net.get_shape())

            net = ops.conv(net, channels=a.ngf*8, kernel=4, stride=2, pad=1, use_bias=True, sn=a.sn, scope='encoder_3')
            net = tf.contrib.layers.instance_norm(net)
            net = tf.nn.relu(net)
            print(net.get_shape())

        with tf.variable_scope("middle"):
            for i in range(a.num_residual_blocks):
                net = ops.resblock_dialated_sn(net, channels=a.ngf*8, rate=2, sn=a.sn, scope='resblock_%d' % i)
    
    with tf.device("/gpu:%d" % (gpu_idx)):
        with tf.variable_scope("decoder"):
            net = ops.upconv(net, channels=a.ngf*4, kernel=3, stride=2, use_bias=True, sn=a.sn, scope='decoder_3')
            net = tf.contrib.layers.instance_norm(net)
            net = tf.nn.relu(net)
            print(net.get_shape())

            net = ops.upconv(net, channels=a.ngf*2, kernel=3, stride=2, use_bias=True, sn=a.sn, scope='decoder_2')
            net = tf.contrib.layers.instance_norm(net)
            net = tf.nn.relu(net)
            print(net.get_shape())

            # self-attention layer
            #net = ops.selfatt(net, condition=tf.image.resize_images(generator_inputs, net.get_shape().as_list()[1:3]), 
            #                input_channel=a.ngf*2, flag_condition=False, channel_fac=a.channel_fac, scope='attention_0')

            net = ops.upconv(net, channels=a.ngf, kernel=3, stride=2, use_bias=True, sn=a.sn, scope='decoder_1')
            net = tf.contrib.layers.instance_norm(net)
            net = tf.nn.relu(net)
            print(net.get_shape())

            #net = ops.selfatt(net, condition=tf.image.resize_images(generator_inputs, net.get_shape().as_list()[1:3]),
            #                input_channel=a.ngf, flag_condition=False, channel_fac=a.channel_fac, scope='attention_1')

            net = ops.conv(net, channels=3, kernel=7, stride=1, pad=3, use_bias=True, sn=a.sn, scope='decoder_0')            
            net = tf.tanh(net)
            print(net.get_shape())

    return net

##################### Discriminators ##############################################
    
def create_discriminator_resgan(discrim_inputs, discrim_targets):
    """
    Discriminator architecture, same as EdgeConnect.
    When SN is True, use_bias is set to False. Dont know why.
    """
    layers = []

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    net = tf.concat([discrim_inputs, discrim_targets], axis=3)

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    net = ops.conv(net, channels=a.ngf, kernel=4, stride=2, pad=1, use_bias=not a.sn, sn=a.sn, scope='discriminator_0')
    net = lrelu(net, 0.2)
    layers.append(net)
    print(net.get_shape())

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf*2]
    net = ops.conv(net, channels=a.ngf*2, kernel=4, stride=2, pad=1, use_bias=not a.sn, sn=a.sn, scope='discriminator_1')
    net = lrelu(net, 0.2)
    layers.append(net)
    print(net.get_shape())

    # layer_3: [batch, 64, 64, ndf*2] => [batch, 32, 32, ndf*4]
    net = ops.conv(net, channels=a.ngf*4, kernel=4, stride=2, pad=1, use_bias=not a.sn, sn=a.sn, scope='discriminator_2')
    net = lrelu(net, 0.2)
    layers.append(net)
    print(net.get_shape())

    # layer_4: [batch, 32, 32, ndf*4] => [batch, 31, 31, ndf*8]
    net = ops.conv(net, channels=a.ngf*8, kernel=4, stride=1, pad=1, use_bias=not a.sn, sn=a.sn, scope='discriminator_3')
    net = lrelu(net, 0.2)
    layers.append(net)
    print(net.get_shape())

    # layer_4: [batch, 31, 31, ndf*4] => [batch, 30, 30, 1]
    net = ops.conv(net, channels=1, kernel=4, stride=1, pad=1, use_bias=not a.sn, sn=a.sn, scope='discriminator_4')
    net = lrelu(net, 0.2)
    layers.append(net)
    print(net.get_shape())

    output = tf.sigmoid(net)

    return output, layers

def create_discriminator_conv(discrim_inputs, discrim_targets):
    n_layers = 3
    layers = []
    
    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    input = tf.concat([discrim_inputs, discrim_targets], axis=3)

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = conv(input, a.ndf, stride=2)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = a.ndf * min(2**(i+1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            convolved = conv(layers[-1], out_channels, stride=stride)
            normalized = batchnorm(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)
            print("conv shape: ", convolved.get_shape())

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = conv(rectified, out_channels=1, stride=1)
        layers.append(convolved)
        output = tf.sigmoid(convolved)
        print("output shape:", output.get_shape())
    return output, layers

def create_discriminator_conv_global(discrim_inputs, discrim_targets):   
    n_layers = 6
    layers = []

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    input = tf.concat([discrim_inputs, discrim_targets], axis=3)

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1_global"):
        convolved = conv(input, a.ndf, stride=2)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 16, 16, ndf * 8]
    # layer_5: [batch, 16, 16, ndf * 8] => [batch, 8,  8,  ndf * 16]
    # layer_6: [batch, 8, 8, ndf * 16] => [batch, 4,  4,  ndf * 16]
    # layer_7: [batch, 4, 4, ndf * 16] => [batch, 2,  2,  ndf * 16]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d_global" % (len(layers) + 1)):
            out_channels = a.ndf * min(2**(i+1), 16)
            stride =  2  # last layer here has stride 1
            convolved = conv(layers[-1], out_channels, stride=stride)
            normalized = batchnorm(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)
            print("g conv shape: ", convolved.get_shape())

    # layer_8: [batch, 2, 2, ndf * 16] => [batch, 1, 1, 1]
    with tf.variable_scope("layer_%d_global" % (len(layers) + 1)):
        convolved = conv(rectified, out_channels=1, stride=1)
        layers.append(convolved)
        output = tf.sigmoid(convolved)
        print("g output shape:", output.get_shape())
        
    return output, layers     

##################### Model #######################################################
    
def create_vgg(images, num_class=1000):
    """
    Build a vgg19 model.
    logits: the output of the vgg19 model
    endpoints: the output of some layers
    """
    with tf.name_scope("vgg_network"):
        logits, endpoints = vgg_19(images, num_classes=1000, global_pool=True)
    return logits, endpoints

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def convert(image):
    if a.aspect_ratio != 1.0:
        # upscale to correct aspect ratio
        size = [a.target_size, int(round(a.target_size * a.aspect_ratio))]
        image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

    return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

def create_tower(inputs, targets, gpu_idx, scope):

    ############### Create Generator ####################################
    with tf.variable_scope("generator") as scope:
        # float32 for TensorFlow
        inputs = tf.cast(inputs, tf.float32)
        targets = tf.cast(targets, tf.float32)
        print("input shape", inputs.get_shape())
        print("targets shape", targets.get_shape())

        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator_resgan(generator_inputs=inputs, generator_outputs_channels=out_channels,gpu_idx=gpu_idx+1)

    ############### Create VGG model for perceptual loss ####################################
    with tf.device("/gpu:%d" % (gpu_idx)):    
        with tf.name_scope("real_vgg") as scope:
            with tf.variable_scope("vgg"):
                real_vgg_logits, real_vgg_endpoints = create_vgg(targets, num_class=a.num_vgg_class)
        with tf.name_scope("fake_vgg") as scope:
            with tf.variable_scope("vgg", reuse=True):
                fake_vgg_logits, fake_vgg_endpoints = create_vgg(targets, num_class=a.num_vgg_class)
    

    ############### Create Discriminator ####################################
    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.device("/gpu:%d" % (gpu_idx)):
        if a.discriminator == "conv":
            print("use conv discriminator")
            create_discriminator = create_discriminator_conv
            create_discriminator_global = create_discriminator_conv_global   
        elif a.discriminator == "resgan":
            print("use resgan discriminator")
            create_discriminator = create_discriminator_resgan
            create_discriminator_global = create_discriminator_conv_global 
        
        ############### Discriminator outputs ###########################
        with tf.name_scope("real_discriminator_patch"):
            with tf.variable_scope("discriminator_patch"):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_real_patch, feature_real_patch = create_discriminator(inputs, targets)
        
        with tf.name_scope("real_discriminator_global"):
            with tf.variable_scope("discriminator_global"):
                # 2x [batch, height, width, channels] => [batch, 1, 1, 1]
                predict_real_global, feature_real_global = create_discriminator_global(inputs, targets)
        
        with tf.name_scope("fake_discriminator"):
            with tf.variable_scope("discriminator_patch", reuse=True):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_fake_patch, feature_fake_patch = create_discriminator(inputs, outputs)
                
        with tf.name_scope("fake_discriminator_global"):
            with tf.variable_scope("discriminator_global", reuse=True):
                # 2x [batch, height, width, channels] => [batch, 1, 1, 1]
                predict_fake_global, feature_fake_global = create_discriminator_global(inputs, outputs)            
    
    ################### Losses #########################################
    with tf.device("/gpu:%d" % (gpu_idx)):
        with tf.name_scope("discriminator_loss"):
            # minimizing -tf.log will try to get inputs to 1
            # predict_real => 1
            # predict_fake => 0
            if a.stabilization == 'lsgan':
                discrim_loss = tf.losses.mean_squared_error(predict_real_patch, tf.ones(predict_real_patch.shape))
                discrim_loss += tf.losses.mean_squared_error(predict_real_global, tf.ones(predict_real_global.shape))
                discrim_loss += tf.losses.mean_squared_error(predict_fake_patch, tf.zeros(predict_fake_patch.shape))
                discrim_loss += tf.losses.mean_squared_error(predict_fake_global, tf.zeros(predict_fake_global.shape))
            else:
                discrim_loss = -tf.reduce_mean(tf.log(predict_real_patch + EPS))
                discrim_loss += -tf.reduce_mean(tf.log(predict_real_global + EPS))
                discrim_loss += -tf.reduce_mean(tf.log(1 - predict_fake_patch + EPS))
                discrim_loss += -tf.reduce_mean(tf.log(1 - predict_fake_global + EPS))
        
        gen_loss = 0
        with tf.name_scope("generator_loss"):
            # predict_fake => 1
            # abs(targets - outputs) => 0
            if a.stabilization == 'lsgan':
                gen_loss_GAN = tf.losses.mean_squared_error(predict_fake_patch, tf.ones(predict_real_patch.shape))
            else:
                gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake_patch + EPS))

            gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
            
            global_step = tf.contrib.framework.get_or_create_global_step()
            l1_weight = tf.train.exponential_decay(a.l1_weight,
                                        global_step,
                                        a.l1_weight_decay_steps,
                                        a.l1_weight_decay_factor,
                                        staircase=True)

            gen_loss += gen_loss_GAN * a.gan_weight
            gen_loss += gen_loss_L1 * l1_weight

        with tf.name_scope("generator_feature_matching_loss"):
            gen_loss_fm = 0
            gen_loss_style = 0
            if a.fm:
                for i in range(a.num_feature_matching):
                    gen_loss_fm += tf.reduce_mean(tf.abs(feature_fake_patch[-i-1] - feature_real_patch[-i-1]))
                gen_loss += gen_loss_fm * a.fm_weight
            if a.style_loss:
                for i in range(a.num_style_loss):
                    gen_loss_style += tf.reduce_mean(tf.abs(ops.gram_matrix(feature_fake_patch[-i-1]) - ops.gram_matrix(feature_real_patch[-i-1])))
                gen_loss += gen_loss_style * a.style_weight
        #loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        #loss_averages_op = loss_averages.apply([gen_loss, gen_loss_GAN, gen_loss_L1, gen_loss_fm, discrim_loss])
        #loss_averages_op = loss_averages.apply([gen_loss, discrim_loss])

        #with tf.control_dependencies([loss_averages_op]):
            #discrim_loss = tf.identity(discrim_loss)
            #gen_loss = tf.identity(gen_loss)
            #gen_loss_GAN = tf.identity(gen_loss_GAN)
            #gen_loss_L1 = tf.identity(gen_loss_L1)
            #gen_loss_fm = tf.identity(gen_loss_fm)

    ############## Summaries ###############################################
    tf.summary.scalar("discriminator_loss", discrim_loss)
    tf.summary.scalar("generator_loss_GAN", gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", gen_loss_L1)
    if a.fm:
        tf.summary.scalar("generator_loss_fm", gen_loss_fm)
    if a.style_loss:
        tf.summary.scalar("generator_loss_style", gen_loss_style)

    inputs_ = deprocess(inputs)
    targets_ = deprocess(targets)
    outputs_ = deprocess(outputs)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs_)
    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets_)
    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs_)


    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)
    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", converted_targets)
    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)
    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(predict_real_patch, dtype=tf.uint8))
    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(predict_fake_patch, dtype=tf.uint8))

    return Tower(inputs=converted_inputs, 
                 targets=converted_targets,
                 outputs=converted_outputs,
                 predict_real=predict_real_patch,
                 predict_fake=predict_fake_patch,
                 discrim_loss=discrim_loss,
                 gen_loss=gen_loss,
                 gen_loss_fm=gen_loss_fm,
                 gen_loss_GAN=gen_loss_GAN,
                 gen_loss_L1=gen_loss_L1)

def create_model_multi_gpu(inputs, targets):
    with tf.name_scope("model"): # tf.Graph().as_default():
        ########## Learning rate #####################################
        global_step = tf.contrib.framework.get_or_create_global_step()
        lr_D = tf.train.exponential_decay(a.lr_discrim,
                                        global_step,
                                        a.lr_decay_steps_D,
                                        a.lr_decay_factor_D,
                                        staircase=True)
        lr_G = tf.train.exponential_decay(a.lr_gen,
                                        global_step,
                                        a.lr_decay_steps_G,
                                        a.lr_decay_factor_G,
                                        staircase=True)

        ########## Calculate the gradients for each model tower ###########
        discrim_grads_tower = []
        gen_grads_tower = []
        discrim_loss_tower = []
        gen_loss_tower = []
        with tf.variable_scope(tf.get_variable_scope()):
            num_towers = math.floor(a.num_gpus/a.num_gpus_per_tower)
            print('number of tower', num_towers)
            for i in range(num_towers):
                with tf.name_scope('tower_%d' % (i)) as scope:
                    # Calculate the loss for one tower of the CIFAR model. This function
                    # constructs the entire CIFAR model but shares the variables across
                    # all towers.
                    tower = create_tower(inputs, targets, i*a.num_gpus_per_tower, scope)

                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    with tf.name_scope("discriminator_train"):
                        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
                        discrim_optim = tf.train.AdamOptimizer(lr_D, a.beta1)
                        discrim_grads_and_vars = discrim_optim.compute_gradients(tower.discrim_loss, var_list=discrim_tvars, colocate_gradients_with_ops=True)
                        #discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)
         
                    with tf.name_scope("generator_train"):
                        gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
                        gen_optim = tf.train.AdamOptimizer(lr_G, a.beta1)
                        gen_grads_and_vars = gen_optim.compute_gradients(tower.gen_loss, var_list=gen_tvars, colocate_gradients_with_ops=True)
                        #gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

                    # Keep track of the gradients across all towers.
                    discrim_grads_tower.append(discrim_grads_and_vars)
                    gen_grads_tower.append(gen_grads_and_vars)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        discrim_grads = average_gradients(discrim_grads_tower)
        gen_grads = average_gradients(gen_grads_tower)

        ################## Train ops #########################################
        discrim_train = discrim_optim.apply_gradients(discrim_grads)
        with tf.control_dependencies([discrim_train]):
            gen_train = gen_optim.apply_gradients(gen_grads)

        #ema = tf.train.ExponentialMovingAverage(decay=0.99)
        #update_losses = ema.apply([discrim_loss, gen_loss])

        incr_global_step = tf.assign(global_step, global_step+1)

    #return tf.group(update_losses, incr_global_step, gen_train), discrim_loss, gen_loss
    return tf.group(incr_global_step, gen_train), tower

def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    mat_dir = os.path.join(a.output_dir, "mats")
    if not os.path.exists(mat_dir):
        os.makedirs(mat_dir)
    
    filesets = []
    if a.load_tfrecord == True:
        for i, fn in enumerate(fetches["filenames"]):
            # int to ascii
            fn = ''.join(chr(n) for n in fn)
            name = fn
            fileset = {"name": name, "step": step}
            # for kind in ["inputs", "outputs", "targets"]:
            # YuhangLi: only save the outputs to save the driver space when tuning hyper-params 
            # YuhangLi: if you want to add "inputs" and "targets" in kinds, please mkdir for them
            for kind in ["outputs"]:
                filename = name + ".png"
                out_path = os.path.join(image_dir, filename)
                contents = fetches[kind][i]
                # images have been converted to png binary and can be saved by only f.write()
                with open(out_path, "wb") as f:
                    f.write(contents)
            # sio.savemat(os.path.join(mat_dir, name+".mat"), {"beta":fetches["beta"][i]})       
            filesets.append(fileset)
    else:
        for i, in_path in enumerate(fetches["paths"]):
            name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
            fileset = {"name": name, "step": step}
            for kind in ["inputs", "outputs", "targets"]:
                filename = name + "-" + kind + ".png"
                if step is not None:
                    filename = "%08d-%s" % (step, filename)
                fileset[kind] = filename
                out_path = os.path.join(image_dir, filename)
                contents = fetches[kind][i]
                with open(out_path, "wb") as f:
                    f.write(contents)
            filesets.append(fileset)
    return filesets

def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for v in fileset:
            index.write("<td><img src='images/%s'></td>" % v)

        index.write("</tr>")
    return index_path

def train():
    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    
    if not os.path.exists(a.output_dir):    
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "export":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)

        # disable these features in test mode
        a.scale_size = a.target_size
        a.flip = False

    # print and save configuration
    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    # read TFRecordDataset
    examples, iterator  = read_tfrecord()
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    # model = create_model(examples.inputs, examples.targets)
    train_op, tower = create_model_multi_gpu(examples.inputs, examples.targets)

    # YuhangLi: only save a part of images for saving driver space.
    num_display_images = 3000
    with tf.name_scope("encode_images"):
        display_fetches = {
            "filenames": examples.filenames,
            "inputs": tf.map_fn(tf.image.encode_png, tower.inputs[:num_display_images], dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, tower.targets[:num_display_images], dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, tower.outputs[:num_display_images], dtype=tf.string, name="output_pngs")
        }

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)
    #    tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)
    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False) # print device placement
    sess_config.gpu_options.allow_growth = True
    with sv.managed_session(config=sess_config) as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        """ Training
        Basic idea of training here is:
            if should run something:
                add it to the fetches
                .
                .
                .
            sess.run(fetches)
                
            if should run something:
                save the result of it
                .
                .
                .
        "Something" includes: metadata, losses, summary, display(save) images
        """
        start = time.time()

        if a.debug:
            a.progress_freq = 1
            a.summary_freq = 1
            a.display_freq = 1
        for step in range(max_steps):
            def should(freq):
                return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

            options = None
            run_metadata = None

            if should(a.trace_freq):
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            fetches = {
                "train": train_op,
                "global_step": sv.global_step,
            }

            if should(a.progress_freq):
                fetches["discrim_loss"] = tower.discrim_loss
                fetches["gen_loss"] = tower.gen_loss
                fetches["gen_loss_GAN"] = tower.gen_loss_GAN
                fetches["gen_loss_L1"] = tower.gen_loss_L1
                if a.fm:
                    fetches["gen_loss_fm"] = tower.gen_loss_fm

            if should(a.summary_freq):
                fetches["summary"] = sv.summary_op

            if should(a.display_freq):
                fetches["display"] = display_fetches

            if a.debug:
                print(step)
            results = sess.run(fetches, options=options, run_metadata=run_metadata)
                
            if should(a.summary_freq):
                print("recording summary")
                sv.summary_writer.add_summary(results["summary"], results["global_step"])

            if should(a.display_freq):
                print("saving display images")
                filesets = save_images(results["display"], step=results["global_step"])
                append_index(filesets, step=True)

            if should(a.trace_freq):
                print("recording trace")
                sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

            if should(a.progress_freq):
                # global_step will have the correct step count if we resume from a checkpoint
                train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                rate = (step + 1) * a.batch_size / (time.time() - start)
                remaining = (max_steps - step) * a.batch_size / rate
                print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                print("discrim_loss", results["discrim_loss"])
                print("gen_loss", results["gen_loss"])
                print("gen_loss_GAN", results["gen_loss_GAN"])
                print("gen_loss_L1", results["gen_loss_L1"])
                if a.fm:
                    print("gen_loss_fm", results["gen_loss_fm"])

            if should(a.save_freq):
                print("saving model")
                saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

            if should(a.evaluate_freq):
                print("evaluating results")
                ## TODO
            if sv.should_stop():
                break
    return

def test():
    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    
    if not os.path.exists(a.output_dir):    
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "export":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)

        # disable these features in test mode
        a.scale_size = a.target_size
        a.flip = False

    # print and save configuration
    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    # read TFRecordDataset
    examples, iterator  = read_tfrecord()
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    # model = create_model(examples.inputs, examples.targets)
    train_op, tower = create_model_multi_gpu(examples.inputs, examples.targets)

    # YuhangLi: only save a part of images for saving driver space.
    num_display_images = 3000
    with tf.name_scope("encode_images"):
        display_fetches = {
            "filenames": examples.filenames,
            "inputs": tf.map_fn(tf.image.encode_png, tower.inputs[:num_display_images], dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, tower.targets[:num_display_images], dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, tower.outputs[:num_display_images], dtype=tf.string, name="output_pngs")
        }

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)
    #    tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    restore_collection =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    saver = tf.train.Saver(var_list=restore_collection, max_to_keep=1)
    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False) # print device placement
    sess_config.gpu_options.allow_growth = True
    with sv.managed_session(config=sess_config) as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            # testing
            # at most, process the test data once
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results, step=step)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                # temporaly commented, error for unknown reason    
                # index_path = append_index(filesets)

            print("wrote index at", index_path)
    return


train()
#test()
