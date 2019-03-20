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
from models import *
from vgg import * 
import ops

slim = tf.contrib.slim
resnet_arg_scope = resnet_utils.resnet_arg_scope

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "test", "export"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=20, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=20, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=50, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")
parser.add_argument("--evaluate_freq", type=int, default=5000, help="evaluate training data every save_freq steps, 0 to disable")

parser.add_argument("--no_hd", dest="hd", action="store_false", help="don't use hd version of CelebA dataset. By default, hd version is used.")
parser.set_defaults(flip=True)
parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--batch_size", type=int, default=8, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=530, help="scale images to this size before cropping to 256x256")
parser.add_argument("--target_size", type=int, default=512, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--random_crop", dest="random_crop", action="store_true", help="crop images randomly")
parser.set_defaults(random_crop=False)
parser.add_argument("--monochrome", dest="monochrome", action="store_true", help="convert image from rgb to gray")
parser.set_defaults(monochrome=False)
parser.add_argument("--lr_gen", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--lr_discrim", type=float, default=0.00002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
parser.add_argument("--fm_weight", type=float, default=1.0, help="weight on feature matching term for generator gradient")
parser.add_argument("--style_weight", type=float, default=1.0, help="weight on style loss term for generator gradient")

#YuhangLi
parser.add_argument("--num_unet", type=int, default=10, help="number of u-connection layers, used only when generator is encoder-decoder")
parser.add_argument("--generator", default="mru", choices=["res", "ir", "ed", "mru", "sa", "sa_I", "resgan"])
parser.add_argument("--discriminator", default="conv", choices=["res", "ir", "conv", "mru", "sa", "sa_I", "resgan"])
parser.add_argument("--input_type", default="df", choices=["edge", "df", "hed"])
parser.add_argument("--double_D", dest="double_D", action="store_true", help="convert image from rgb to gray")
parser.set_defaults(double_D=True)
parser.add_argument("--load_image", dest="load_tfrecord", action="store_false", help="if true, read dataset from TFRecord, otherwise from images")
parser.set_defaults(load_tfrecord=True)
parser.add_argument("--num_examples", required=True, type=int, help="number of training/testing examples in TFRecords. required, since TFRecords do not have metadata")
parser.add_argument("--channel_fac", default=16, type=int, help="faction of channel in self attention modual. Set to large to save GPU memory")
parser.add_argument("--enc_atten", type=str, default="FTFFF")
parser.add_argument("--dec_atten", type=str, default="FFFTF")
parser.add_argument("--no_sn", dest="sn", action="store_false", help="do not use spectral normalization")
parser.set_defaults(sn=True)
parser.add_argument("--no_fm", dest="fm", action="store_false", help="do not use feature matching loss")
parser.set_defaults(fm=True)
parser.add_argument("--no_style_loss", dest="style_loss", action="store_false", help="do not use style loss")
parser.set_defaults(style_loss=True)
parser.add_argument("--residual_blocks", type=int, default=8, help="number of residual blocks in resgan generator")
parser.add_argument("--num_feature_matching", type=int, default=3, help="number of layers in feature matching loss, count from the last layer of the discriminator")
parser.add_argument("--num_style_loss", type=int, default=3, help="number of layers in style loss, count from the last layer of the discriminator")
parser.add_argument("--num_vgg_class", type=int, default=1000, help="number of class of pretrained vgg network")
parser.add_argument("--num_gpus", type=int, default=4, help="number of GPUs used for training")
parser.add_argument("--num_gpus_per_tower", type=int, default=2, help="number of GPUs per tower used for training")
parser.add_argument("--lr_decay_steps_D", type=int, default=10000, help="learning rate decay steps for discriminator")
parser.add_argument("--lr_decay_steps_G", type=int, default=10000, help="learning rate decay steps for generator")
parser.add_argument("--lr_decay_factor_D", type=float, default=0.1, help="learning rate decay factor for discriminator")
parser.add_argument("--lr_decay_factor_G", type=float, default=0.1, help="learning rate decay factor for generator")
parser.add_argument("--df_norm_value", type=float, default=64.0, help="the nomalizaiton value of distance fields")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
a = parser.parse_args()

EPS = 1e-12

NUM_SAVE_IMAGE = 100


Examples = collections.namedtuple("Examples", "filenames, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, beta_list, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_loss_fm, gen_grads_and_vars, train")
seed = random.randint(0, 2**31 - 1)

##################### Data #####################################################

def transform(image):
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
    if a.flip:
        r = tf.image.random_flip_left_right(r, seed=seed)
    if a.monochrome:
        r = tf.image.rgb_to_grayscale(r)
    if not height == width:
        # center crop to correct ratio
        size = min(height, width)
        oh = (height - size) // 2
        ow = (width - size) // 2
        r = tf.image.crop_to_bounding_box(image=r, offset_height=oh, offset_width=ow, target_height=size, target_width=size)
    if  a.random_crop: 
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
        df = df/a.df_norm_value # normalize the distance fields, by a given value, to fit grayscale
        df = (df) * 2. - 1.    
        df = transform(tf.image.grayscale_to_rgb(df))
        condition = df

    elif a.input_type == "edge": 
        edge = tf.decode_raw(parsed_features['edge'], tf.float32) 
        edge = tf.reshape(edge, [512, 512, 1])
        edge = (edge) * 2. - 1.
        edge = transform(tf.image.grayscale_to_rgb(edge))
        condition = edge

    elif a.input_type == "hed": 
        hed = tf.decode_raw(parsed_features['hed'], tf.float32) 
        hed = tf.reshape(hed, [512, 512, 1])
        hed = (hed) * 2. - 1.
        hed = transform(tf.image.grayscale_to_rgb(hed))
        condition = hed

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
            for i in range(a.residual_blocks):
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

def create_generator_selfatt_stack(generator_inputs, generator_outputs_channels, flag_I=True):
    """
    Replace conv in encoder-decoder network with MRU.
    First and last layer still use conv and deconv.
    No dropout presently.
    Stride = 2, output_channel = input_channel * 2 
    """
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = conv(generator_inputs, a.ngf, stride=2)
        rectified = lrelu(output, 0.2)
        layers.append(output)

    layer_specs = [
        (a.ngf * 2), # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        (a.ngf * 4), # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        (a.ngf * 8), # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        (a.ngf * 8), # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        (a.ngf * 8), # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        # a.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        # a.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        ]
    
    beta_list = []
    
    for i, out_channels in enumerate(layer_specs):
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]           
            convolved = conv(layers[-1], out_channels, stride=2)
            output = batchnorm(convolved)
            output = lrelu(output, 0.2)
            if a.enc_atten[i]=='T':
                output, beta = selfatt(output, tf.image.resize_images(generator_inputs, output.shape[1:3]), out_channels, flag_I=flag_I, channel_fac=a.channel_fac)
                beta_list.append(beta)
            layers.append(output)
            
    with tf.device("/gpu:1"):   
        layer_specs = [
            # (a.ngf * 8, 0.0),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
            # (a.ngf * 8, 0.0),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (a.ngf * 8, 0.0),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
            ]
        num_encoder_layers = len(layers)
            
        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                if decoder_layer == 0 or decoder_layer >= a.num_unet:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer
                    input = layers[-1]
                else:               
                    input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

                # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                output = ops.up_sample(input, scale_factor=2)
                output = ops.conv(output, channels=out_channels, kernel=3, stride=1, pad=1, sn=True) 
                output = ops.batch_norm(output)
                output = ops.lrelu(output)
                if a.dec_atten[i]=='T':
                    output, beta = selfatt(output, tf.image.resize_images(generator_inputs, output.shape[1:3]), out_channels, flag_I=flag_I, channel_fac=a.channel_fac)                    
                    beta_list.append(beta)
                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)
                layers.append(output)
                
    with tf.device("/gpu:1"): 
        # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
        with tf.variable_scope("decoder_1"):
            input = tf.concat([layers[-1], layers[0]], axis=3)
            rectified = tf.nn.relu(input)
            output = deconv(rectified, generator_outputs_channels)
            output = tf.tanh(output)
            layers.append(output)

    return layers[-1], beta_list

def create_generator_selfatt(generator_inputs, generator_outputs_channels, flag_I=True):
    """
    Add Conditional Self-Attention Modual to the U-Net Generator.
    By default, 256x256 => 256x256
    
    Args:
    generator_inputs: a tensor of input images, [b, h, w, n], with each pixel value [-1, 1].
    generator_outputs_channels: the number of generator output channels.
    flag_I: bool flag to indicate if add conditional input to self-attention layer.
    
    Returns:
    layers[-1]: the output of generator, eg the generated images batch, [b, h, w, n], with each pixel value [-1, 1].
    beta_list: list of beta matrics, save to visualize attention maps.  

    Note: a beta matrix is too large to view directly, visualize it row by row as attention maps
    """
    # save output of layers for skip connections
    layers = []
    ###################### encoder ###########################################
    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = ops.conv(generator_inputs, channels=a.ngf, kernel=4, stride=2, pad=1, sn=a.sn)
        output = ops.lrelu(output, 0.2)
        # consider: append output before/after lrelu.
        # Why not use batch norm in the first layer?
        layers.append(output)

    # encoder information, (out_channels)
    encoder_layers = [
        (a.ngf * 2), # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        (a.ngf * 4), # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        (a.ngf * 8), # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        (a.ngf * 8), # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        (a.ngf * 8), # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        # a.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        # a.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        ]
        
    beta_list = []
    
    for i, out_channels in enumerate(encoder_layers):
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]  
            # Conv + BN + leakyReLU + [selfatt]
            output = ops.conv(layers[-1], channels=out_channels, kernel=4, stride=2, pad=1, sn=a.sn)
            output = batchnorm(output) # not use ops.batch_norm, because do not know its update strategy
            output = ops.lrelu(output, 0.2)
            if a.enc_atten[i]=='T':
                output, beta = selfatt(output, tf.image.resize_images(generator_inputs, output.shape[1:3]), out_channels, flag_I=flag_I, channel_fac=a.channel_fac)
                beta_list.append(beta)
            layers.append(output)
    
    ###################### decoder ###########################################
    # Explictly assign decoder to /gpu:1
    # Consider: layers[] is assign to /gpu:0 by default, skip connections involve communication between GPUs.
    with tf.device("/gpu:1"):   
        # decoder information: (out_channels, dropout rate)
        decoder_layers = [
            # (a.ngf * 8, 0.0),     # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
            # (a.ngf * 8, 0.0),     # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (a.ngf * 8, 0.0),       # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (a.ngf * 8, 0.0),       # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (a.ngf * 4, 0.0),       # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (a.ngf * 2, 0.0),       # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (a.ngf, 0.0),           # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
            ]
        num_encoder_layers = len(layers)
            
        for decoder_layer, (out_channels, dropout) in enumerate(decoder_layers):
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                if decoder_layer == 0 or decoder_layer >= a.num_unet:
                    # first decoder layer is directly connected to the skip_layer
                    # a.num_unet controls the number of skip connections
                    input = layers[-1]
                else:               
                    input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

                # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                # Up-sample + 1x1 Conv + BN + leakyReLU + [selfatt] + [dropout]
                output = ops.up_sample(input, scale_factor=2) #use upsample+conv replace deconv to advoid checkboard effect
                output = ops.conv(output, channels=out_channels, kernel=3, stride=1, pad=1, sn=True) 
                output = batchnorm(output)
                output = ops.lrelu(output)
                if a.dec_atten[i]=='T':
                    output, beta = selfatt(output, tf.image.resize_images(generator_inputs, output.shape[1:3]), out_channels, flag_I=flag_I, channel_fac=a.channel_fac)                    
                    beta_list.append(beta)
                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)
                layers.append(output)
                
    with tf.device("/gpu:1"): 
        # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
        with tf.variable_scope("decoder_1"):
            output = tf.concat([layers[-1], layers[0]], axis=3)
            output = tf.nn.relu(output)
            output = deconv(output, generator_outputs_channels)
            output = tf.tanh(output)
            layers.append(output)

    return layers[-1], beta_list

def create_generator_mru(generator_inputs, generator_outputs_channels):
    """
    Replace conv in encoder-decoder network with MRU.
    First and last layer still use conv and deconv.
    No dropout presently.
    Stride = 2, output_channel = input_channel * 2 """
    
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = conv(generator_inputs, a.ngf, stride=2)
        rectified = lrelu(output, 0.2)

        layers.append(output)

    layer_specs = [
        a.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        a.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        # a.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        # a.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            output = mru(layers[-1], tf.image.resize_images(generator_inputs, layers[-1].shape[1:3]), out_channels, stride=2)
            layers.append(output)

    layer_specs = [
        # (a.ngf * 8, 0.0),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        # (a.ngf * 8, 0.0),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (a.ngf * 8, 0.0),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]
    
   

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0 or decoder_layer >= a.num_unet:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = demru(input,  tf.image.resize_images(generator_inputs, input.shape[1:3]), out_channels, stride=2)
            
            
            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]
    return

def create_generator_resnet(generator_inputs, generator_outputs_channels):
    with tf.variable_scope('Generator'):
        shortcut = slim.conv2d(generator_inputs, a.ngf, [1, 1], stride=1,
                             normalizer_fn=None, activation_fn=tf.nn.relu)
        # seperate one bottleneck operation in case of the scope problem
        residual = bottleneck(shortcut , depth=a.ngf, depth_bottleneck=a.ngf, stride=1)                  
        residual = slim.repeat(residual, 10, bottleneck, depth=a.ngf, depth_bottleneck=a.ngf, stride=1)
        residual = slim.conv2d(residual,  a.ngf, [1, 1], stride=1,
                           normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        net = shortcut + residual
        
        net = slim.conv2d(net, a.ngf, [1, 1], stride=1,
                           normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        net = slim.conv2d(net, a.ngf, [1, 1], stride=1,
                           normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        output = slim.conv2d(net, generator_outputs_channels, [1, 1], stride=1,
                           normalizer_fn=None, activation_fn=None)                             
    return output

def create_generator_ed(generator_inputs, generator_outputs_channels):
    """create a generator in encoder-decoder architecture with u-connection"""
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = conv(generator_inputs, a.ngf, stride=2)
        layers.append(output)

    layer_specs = [
        a.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        a.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        a.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        a.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(rectified, out_channels, stride=2)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (a.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0 or decoder_layer >= a.num_unet:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]   

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

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = conv(rectified, out_channels=1, stride=1)
        layers.append(convolved)
        output = tf.sigmoid(convolved)
       
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
    # layer_6: [batch, 8, 8, ndf * 16] => [batch, 4,  4,  ndf * 32]
    # layer_7: [batch, 4, 4, ndf * 32] => [batch, 2,  2,  ndf * 64]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d_global" % (len(layers) + 1)):
            out_channels = a.ndf * 2**(i+1)
            stride =  2  # last layer here has stride 1
            convolved = conv(layers[-1], out_channels, stride=stride)
            normalized = batchnorm(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

    # layer_8: [batch, 2, 2, ndf * 64] => [batch, 1, 1, 1]
    with tf.variable_scope("layer_%d_global" % (len(layers) + 1)):
        convolved = conv(rectified, out_channels=1, stride=1)
        layers.append(convolved)
        output = tf.sigmoid(convolved)
        
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

def create_model(inputs, targets):
    #with tf.device("/gpu:1"):
    with tf.variable_scope("generator") as scope:
        # float32 for TensorFlow
        inputs = tf.cast(inputs, tf.float32)
        targets = tf.cast(targets, tf.float32)
        out_channels = int(targets.get_shape()[-1])
        if a.generator == 'res':
            outputs = create_generator_resnet(inputs, out_channels)
            beta_list = []
        elif a.generator == 'ir':
            outputs = create_generator_irnet(inputs, out_channels)
            beta_list = []
        elif a.generator == 'ed':
            outputs = create_generator_ed(inputs, out_channels)
            beta_list = []
        elif a.generator == 'mru':
            outputs = create_generator_mru(inputs, out_channels)
            beta_list = []
        elif a.generator == 'sa':
            outputs, beta_list = create_generator_selfatt(inputs, out_channels, flag_I=False)
        elif a.generator == 'sa_I':
            outputs, beta_list = create_generator_selfatt(inputs, out_channels)
        elif a.generator == 'resgan':
            outputs = create_generator_resgan(inputs, out_channels)

    with tf.device("/gpu:0"):    
        with tf.name_scope("real_vgg") as scope:
            with tf.variable_scope("vgg"):
                real_vgg_logits, real_vgg_endpoints = create_vgg(targets, num_class=a.num_vgg_class)
        with tf.name_scope("fake_vgg") as scope:
            with tf.variable_scope("vgg", reuse=True):
                real_vgg_logits, real_vgg_endpoints = create_vgg(targets, num_class=a.num_vgg_class)
    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.device("/gpu:0"):
        if a.discriminator == "conv":
            create_discriminator = create_discriminator_conv
            create_discriminator_global = create_discriminator_conv_global   
        elif a.discriminator == "resgan":
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
        
        ################### Loss #########################################
        with tf.name_scope("discriminator_loss"):
            # minimizing -tf.log will try to get inputs to 1
            # predict_real => 1
            # predict_fake => 0
            discrim_loss = tf.reduce_mean(-( \
                tf.log(predict_real_patch + EPS) \
                + tf.log(predict_real_global + EPS) \
                + tf.log(1 - predict_fake_patch + EPS) \
                + tf.log(1 - predict_fake_global + EPS) \
                ))
        
        gen_loss = 0
        with tf.name_scope("generator_loss"):
            # predict_fake => 1
            # abs(targets - outputs) => 0
            gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake_patch + EPS))
            gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
            gen_loss += gen_loss_GAN * a.gan_weight
            gen_loss += gen_loss_L1 * a.l1_weight

        with tf.name_scope("generator_feature_matching_loss"):
            gen_loss_fm = 0
            if a.fm:
                for i in range(a.num_feature_matching):
                    gen_loss_fm += tf.reduce_mean(tf.abs(feature_fake_patch[-i-1] - feature_real_patch[-i-1]))
                gen_loss += gen_loss_fm * a.fm_weight

        ################## Train ops #########################################
        with tf.name_scope("discriminator_train"):
            discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
            discrim_optim = tf.train.AdamOptimizer(a.lr_discrim, a.beta1)
            discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars, colocate_gradients_with_ops=True)
            discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)
         
        with tf.name_scope("generator_train"):
            with tf.control_dependencies([discrim_train]):
                gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
                gen_optim = tf.train.AdamOptimizer(a.lr_gen, a.beta1)
                gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars, colocate_gradients_with_ops=True)
                gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        update_losses = ema.apply([discrim_loss, gen_loss, gen_loss_GAN, gen_loss_fm, gen_loss_L1])

        global_step = tf.contrib.framework.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real_patch,
        predict_fake=predict_fake_patch,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_loss_fm=ema.average(gen_loss_fm),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        # beta_list=beta_list,
        beta_list=None,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )

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

def main():
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

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    if a.mode == "export":
        # export the generator to a meta graph that can be imported later for standalone generation
        input = tf.placeholder(tf.string, shape=[1])
        input_data = tf.decode_base64(input[0])
        input_image = tf.image.decode_png(input_data)

        # remove alpha channel if present
        input_image = tf.cond(tf.equal(tf.shape(input_image)[2], 4), lambda: input_image[:,:,:3], lambda: input_image)
        # convert grayscale to RGB
        input_image = tf.cond(tf.equal(tf.shape(input_image)[2], 1), lambda: tf.image.grayscale_to_rgb(input_image), lambda: input_image)

        input_image = tf.image.convert_image_dtype(input_image, dtype=tf.float32)
        input_image.set_shape([a.target_size, a.target_size, 3])
        batch_input = tf.expand_dims(input_image, axis=0)

        with tf.variable_scope("generator"):
            batch_output = deprocess(create_generator(preprocess(batch_input), 3))

        output_image = tf.image.convert_image_dtype(batch_output, dtype=tf.uint8)[0]
        if a.output_filetype == "png":
            output_data = tf.image.encode_png(output_image)
        elif a.output_filetype == "jpeg":
            output_data = tf.image.encode_jpeg(output_image, quality=80)
        else:
            raise Exception("invalid filetype")
        output = tf.convert_to_tensor([tf.encode_base64(output_data)])

        key = tf.placeholder(tf.string, shape=[1])
        inputs = {
            "key": key.name,
            "input": input.name
        }
        tf.add_to_collection("inputs", json.dumps(inputs))
        outputs = {
            "key":  tf.identity(key).name,
            "output": output.name,
        }
        tf.add_to_collection("outputs", json.dumps(outputs))

        init_op = tf.global_variables_initializer()
        restore_saver = tf.train.Saver()
        export_saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init_op)
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            restore_saver.restore(sess, checkpoint)
            print("exporting model")
            export_saver.export_meta_graph(filename=os.path.join(a.output_dir, "export.meta"))
            export_saver.save(sess, os.path.join(a.output_dir, "export"), write_meta_graph=False)

        return

    # read TFRecordDataset
    
    examples, iterator  = read_tfrecord()
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputs, examples.targets)

    # undo colorization splitting on images that we use for display/output

    inputs = deprocess(examples.inputs)
    targets = deprocess(examples.targets)
    outputs = deprocess(model.outputs)

    def convert(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [a.target_size, int(round(a.target_size * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)
    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)
    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)
    
    #print("length of beta list..................", len(model.beta_list))
    
    if 0:#a.generator=='sa' or a.generator=='sa_I':
        # convert beta matrices to image to show as attention maps
        with tf.name_scope("convert_beta"):
            converted_betas = tf.image.convert_image_dtype(model.beta_list[-1], dtype=tf.uint8, saturate=True)

    # YuhangLi: only save a part of images for saving driver space.
    num_display_images = 3000
    with tf.name_scope("encode_images"):
        display_fetches = {
            "filenames": examples.filenames,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs[:num_display_images], dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets[:num_display_images], dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs[:num_display_images], dtype=tf.string, name="output_pngs")
        }

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)
    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", converted_targets)
    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)
        
    if 0: #a.generator=='sa' or a.generator=='sa_I': 
        with tf.name_scope("betas_summary"):
            tf.summary.image("betas", tf.image.convert_image_dtype(model.beta_list[-1], dtype=tf.uint8, saturate=True))
            
    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))
    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)
    tf.summary.scalar("generator_loss_fm", model.gen_loss_fm)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)
    #    tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)
    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    sess_config = tf.ConfigProto(allow_soft_placement=True)
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
        else:
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

            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN
                    fetches["gen_loss_L1"] = model.gen_loss_L1
                    fetches["gen_loss_fm"] = model.gen_loss_fm

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches

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
                    print("gen_loss_GAN", results["gen_loss_GAN"])
                    print("gen_loss_L1", results["gen_loss_L1"])
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


main()