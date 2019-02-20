from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import functools
import random
import collections
import math
import time
import scipy.ndimage as sn 
import scipy.io as sio
import shutil

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
parser.add_argument("--summary_freq", type=int, default=99, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=45, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=999, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=4999, help="save model every save_freq steps, 0 to disable")
parser.add_argument("--evaluate_freq", type=int, default=4999, help="evaluate training data every save_freq steps, 0 to disable")

parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--batch_size", type=int, default=8, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--target_size", type=int, default=224, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--no_random_crop", dest="random_crop", action="store_false", help="don't crop images randomly")
parser.set_defaults(random_crop=True)
parser.add_argument("--monochrome", dest="monochrome", action="store_true", help="convert image from rgb to gray")
parser.set_defaults(monochrome=False)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

#YuhangLi
parser.add_argument("--num_unet", type=int, default=10, help="number of u-connection layers, used only when generator is encoder-decoder")
parser.add_argument("--generator", default="atte", choices=["res", "ir", "ed", "atte", "sa", "sa_I"])
parser.add_argument("--discriminator", default="conv", choices=["res", "ir", "conv", "atte", "sa", "sa_I"])
parser.add_argument("--input_type", default="df", choices=["edge", "df"])
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

# Disentangle
parser.add_argument("--lambda_para", type=float, default=0.1, help="lambda parameter of loss_gen_reconstruction")
parser.add_argument("--num_identity", type=int, default=1000, help="number of identity")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
a = parser.parse_args()

EPS = 1e-12
NUM_SAVE_IMAGE = 100

Examples = collections.namedtuple("Examples", "images, labels, steps_per_epoch")
Model = collections.namedtuple("Model", "identity_inputs, attribute_inputs, \
                                            predict_real, \
                                            predict_fake, \
                                            loss_identity, \
                                            grads_and_vars_identity, \
                                            loss_classifier, \
                                            grads_and_vars_classifier, \
                                            loss_discrim, \
                                            grads_and_vars_discrim, \
                                            loss_gen_reconstruction, \
                                            loss_gen_classifer, \
                                            loss_gen_discriminator, \
                                            loss_gen, \
                                            accuracy_identity, \
                                            accuracy_classifier, \
                                            grads_and_vars_gen,\
                                            loss_gen_GAN,\
                                            grads_and_vars_attribute,\
                                            outputs,\
                                            train, \
                                            onehot_identities, \
                                            probability_c_xs, \
                                            probability_c_outputs, \
         ")
seed = random.randint(0, 2**31 - 1)

##################### Functions ###############################################
def convert(image):
    if a.aspect_ratio != 1.0:
        # upscale to correct aspect ratio
        size = [a.target_size, int(round(a.target_size * a.aspect_ratio))]
        image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

    return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

##################### Data Reader/Saver #######################################

def transform(image, seed):
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
    print(height, 'x x x', width)
    print(r.get_shape())
    if a.flip:
        r = tf.image.random_flip_left_right(r, seed=seed)
    if not 1: #height == width:
        # center crop to correct ratio
        print("height not equal to width")
        size = min(height, width)
        oh = (height - size) // 2
        ow = (width - size) // 2
        print(ow, "oooooo", oh)
        r = tf.image.crop_to_bounding_box(image=r, offset_height=oh, offset_width=ow, target_height=size, target_width=size)
    
    # resize to a.target_size
    r = tf.image.resize_images(r, [a.target_size, a.target_size], method=tf.image.ResizeMethod.AREA)

    return r 

def parse_function(example_proto):
    '''
        
    '''            
    features = {
                'height':  tf.FixedLenFeature([], tf.int64),
                'width':  tf.FixedLenFeature([], tf.int64),
                'depth':  tf.FixedLenFeature([], tf.int64),
                'image':  tf.FixedLenFeature([], tf.string),    
                'label':  tf.FixedLenFeature([], tf.int64),
            }        
    
    parsed_features = tf.parse_single_example(example_proto, features=features) 
    
    image = tf.decode_raw(parsed_features['image'], tf.uint8)
    height = tf.cast(parsed_features['height'], tf.int32)
    width = tf.cast(parsed_features['width'], tf.int32)
    label = tf.cast(parsed_features['label'], tf.int32)
    image = tf.reshape(image, [height, width, 3])
    image = tf.image.convert_image_dtype(image, dtype=tf.float64)
    image = image * 2. -1.    
    
    seed = random.randint(0, 2**31 - 1)
    image = transform(image, seed) 

    return image, label

def read_tfrecord():
    tfrecord_fn = glob.glob(os.path.join(a.input_dir, "*.tfrecords"))
    dataset = tf.data.TFRecordDataset(tfrecord_fn)
    dataset = dataset.map(parse_function)  # Parse the record into tensors. 
    dataset = dataset.repeat()  # Repeat the input indefinitely.
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(a.batch_size)
    iterator = dataset.make_one_shot_iterator()

    image, label = iterator.get_next()
    image.set_shape([a.batch_size, a.target_size, a.target_size, 3])

    steps_per_epoch = int(math.ceil(a.num_examples / a.batch_size))
    
    # show read results for code test
    #sess = tf.Session()
    #image, label, path = sess.run(iterator.get_next())
    #Image.fromarray(image[0,:,:,:], "RGB").save('./a1.jpg')
    #print(label, path)
    
    return Examples(
        #paths=None,
        images=image,
        labels=label,
        steps_per_epoch=steps_per_epoch
    ), iterator

def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    filesets = []
    return 
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

##################### Encoders ###############################################
def create_encoder_conv(encoder_inputs, output_channels):
    """create a basic convolutional encoder
       [batch, 256, 256, in_channels] =>[batch, 1, 1, output_channels]

       Returns:
        convolved: as the feature of the inputs
        output: output of the encoder, softmaxed
    """
    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = conv(encoder_inputs, a.ngf, stride=2)

    layer_specs = [
        a.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        a.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        a.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        output_channels, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, output_channels]
    ]

    i = 2
    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (i)):
            rectified = lrelu(output, 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(rectified, out_channels, stride=2)
            output = tf.nn.softmax(convolved)
            i += 1
    
    return convolved, output

def create_encoder_vgg(encoder_inputs, output_channels):
    logits, endpoints = vgg_16(encoder_inputs, num_classes=output_channels, spatial_squeeze=True, global_pool=True) # set num_class to get logits before fc7
    return logits, endpoints['global_pool']

##################### Decoders ###############################################
def create_decoder_conv(deconder_inputs, generator_outputs_channels=3):
    """create a basic convolutional decoder
        [batch, 1, 1, in_channels] => [batch, 256, 256, generator_outputs_channels]
    """
    layer_specs = [
        (a.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (a.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]
    i = 8
    output = deconder_inputs
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):       
        with tf.variable_scope("decoder_%d" % (i)):
            i -= 1
            rectified = tf.nn.relu(output)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = deconv(rectified, out_channels)
            output = batchnorm(output)
            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        rectified = tf.nn.relu(output)
        output = deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)

    return output

def create_decoder_vgg(deconder_inputs, generator_outputs_channels=3):
    """
    Reverse the vgg net.
    """
    net = slim.conv2d(deconder_inputs, 4096, [1, 1], scope='fc7') # [bs, 1, 1, 8192] => [bs, 1, 1, 4096]
    # TODO: consider relu
    with tf.variable_scope("deconv_1"):
        net = deconv(net, 512, 3, 7) # [bs, 1, 1, 4096] => [bs, 7, 7, 512]
        net = batchnorm(net)
    with tf.variable_scope("deconv_2"):
        net = deconv(net, 512, 3, 2) # [bs, 7, 7, 512] => [bs, 14, 14, 512]
        net = batchnorm(net)
    with tf.variable_scope("deconv_3"):
        net = deconv(net, 256, 3, 2) # [bs, 14, 14, 512] => [bs, 28, 28, 256]
        net = batchnorm(net)
    with tf.variable_scope("deconv_4"):
        net = deconv(net, 128, 3, 2) # [bs, 28, 28, 512] => [bs, 56, 56, 128]
        net = batchnorm(net)
    with tf.variable_scope("deconv_5"):
        net = deconv(net, 64, 3, 2) # [bs, 56, 56 128] => [bs, 112, 112, 64]
        net = batchnorm(net)
    with tf.variable_scope("deconv_6"):
        net = deconv(net, 3, 3, 2) # [bs, 112, 112, 64] => [bs, 224, 224, 3]
        net = tf.tanh(net)
    return net

##################### Discriminators #########################################
def create_discriminator_unconditional(discrim_inputs):
    n_layers = 3
    layers = []

    # layer_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = conv(discrim_inputs, a.ndf, stride=2)
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
        output = tf.sigmoid(convolved)
        layers.append(output)

    return convolved, layers[-1]    

def create_discriminator_vgg(discrim_inputs):
    logits, endpoints = vgg_16(discrim_inputs, num_classes=2, spatial_squeeze=True, global_pool=True) # set num_class to get logits before fc7
    print(endpoints)
    return endpoints['global_pool'],tf.sigmoid(endpoints['pool5'])
##################### Model ##################################################
def create_model(identity_images, attribute_images, identity_indices, step, class_number):
    """
    identity_inputs, attribute_input: two inputs of the model, same or different, [B, 300, 300, 3]
    indetity_indices: the idetity indices of face images, [B, 1]
    class_number: the number of indentities in the whole dataset
    flag: indicated whether the identity_inputs and the attribute are the same (true) or different (false)
    """
    create_discriminator = create_discriminator_unconditional
    create_encoder = create_encoder_vgg
    create_decoder = 1

    identity_inputs = tf.get_variable("identity_inputs", [a.batch_size, a.target_size, a.target_size, 3],initializer=tf.zeros_initializer())
    attribute_inputs = tf.get_variable("attribute_inputs", [a.batch_size, a.target_size, a.target_size, 3],initializer=tf.zeros_initializer())

    identity_inputs=(identity_images)

    # select attribute inputs
    condition_flag = tf.equal(tf.mod(step, 2), 1)
    attribute_inputs = tf.cond(condition_flag, lambda: identity_images, lambda: attribute_images)

    onehot_identities = tf.one_hot(identity_indices, class_number)
    with tf.device("/gpu:1"):
        identity_inputs = tf.cast(identity_inputs, tf.float32)
        attribute_inputs = tf.cast(attribute_inputs, tf.float32)
        out_channels = int(identity_inputs.get_shape()[-1])
        with tf.name_scope("identity_network"):
            with tf.variable_scope("encoder"):
                logits_I, f_I  = create_encoder(identity_inputs,class_number)
                
    with tf.device("/gpu:1"):           
        with tf.variable_scope("attribute_network"):
            _, f_A  = create_encoder(attribute_inputs,None)
        # TODO: f_A_I = [f_I^T, f_A^T]^T, should be modified.
        f_A_I = tf.concat([f_I, f_A], axis=3)

    with tf.device("/gpu:0"):
        with tf.variable_scope("generator_network"):
            gen_outputs = create_decoder(f_A_I, out_channels)

    with tf.device("/gpu:1"):
        with tf.name_scope("classifier_network"):
            with tf.variable_scope("encoder", reuse=True):
                logits_C, f_C = create_encoder(gen_outputs,class_number)


    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.device("/gpu:0"):     
        with tf.name_scope("real_discriminator"):
            with tf.variable_scope("discriminator"):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                f_D_attribute, predict_real = create_discriminator(attribute_inputs)

        with tf.name_scope("fake_discriminator"):
            with tf.variable_scope("discriminator", reuse=True):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                f_D_outputs,predict_fake = create_discriminator(gen_outputs)
                                
       ########## Losses ##############################################
    with tf.device("/gpu:1"):
        with tf.name_scope("identity_loss"):
            # min -E[logP(c|x^s)] = min E[log (Vector_I - P(c|x^s))], where Vector_I is one hot identity vector
            loss_identity = tf.losses.softmax_cross_entropy(onehot_identities, logits_I)
            accuracy_identity =  tf.metrics.accuracy(labels=identity_indices, predictions=tf.argmax(logits_I, axis=1))
            #clip_soft_logit_I = tf.clip_by_value(tf.nn.softmax(logits_I), 1e-10, 0.999999999)
            #loss_identity = -tf.reduce_mean(tf.reduce_sum(onehot_identities * tf.log(clip_soft_logit_I) \
            #    + (1 - onehot_identities) * tf.log(1 - clip_soft_logit_I)))
    with tf.device("/gpu:1"):
        with tf.name_scope("classifier_loss"):
            # min -E[logP(c|x')] = min E[log (Vector_I - P(c|x'))], where Vector_I is one hot identity vector
            loss_classifier = tf.losses.softmax_cross_entropy(onehot_labels=onehot_identities, logits=logits_C)
            accuracy_classifier =  tf.metrics.accuracy(labels=identity_indices, predictions=tf.argmax(logits_C, axis=1))

            #clip_soft_logit_C = tf.clip_by_value(tf.nn.softmax(logits_C), 1e-10, 0.999999999)
            #loss_classifier = -tf.reduce_mean(tf.reduce_sum(onehot_identities * tf.log(clip_soft_logit_C) \
            #    + (1 - onehot_identities) * tf.log(1 - clip_soft_logit_C)))
        
    with tf.device("/gpu:0"):
        with tf.name_scope("discriminator_loss"):
            # minimizing -tf.log will try to get inputs to 1
            # predict_real => 1
            # predict_fake => 0
            loss_discrim = tf.reduce_mean(-(tf.log(predict_real + EPS) \
                + tf.log(1 - predict_fake + EPS)))
        ## Generator losses ####
        # L_G
    with tf.device("/gpu:1"):
        with tf.name_scope("generator_loss"):
            # predict_fake => 1
            # abs(targets - outputs) => 0
            loss_gen_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))

        # L_GD
    with tf.device("/gpu:0"):
        with tf.name_scope("generator_discriminator_feature_loss"):
            loss_gen_discriminator = 0.5 * tf.norm(f_D_attribute -  f_D_outputs)

        # L_GC
    with tf.device("/gpu:0"):
        with tf.name_scope("generator_classifier_feature_loss"):
            loss_gen_classifer = 0.5 * tf.norm(f_I - f_C)
        
        # L_GR
    with tf.device("/gpu:0"):
        with tf.name_scope("reconstruction_loss"):
            if step % 2 == 1:
                lambda_para = 1.0
            else:
                lambda_para = a.lambda_para

            loss_gen_reconstruction = 0.5 * lambda_para * tf.norm(attribute_inputs - gen_outputs)

        ############# Train operators #############################################
    with tf.device("/gpu:1"):        
        with tf.name_scope("identity_network_train"):
            identity_tvars = [var for var in tf.trainable_variables() if var.name.startswith("encoder")]
            identity_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            grads_and_vars_identity = identity_optim.compute_gradients(loss_identity, var_list=identity_tvars, colocate_gradients_with_ops=True)
            identity_train = identity_optim.apply_gradients(grads_and_vars_identity)

    with tf.device("/gpu:1"):
        with tf.name_scope("classifier_network_train"):
            classifier_tvars = [var for var in tf.trainable_variables() if var.name.startswith("encoder")]
            classifier_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            grads_and_vars_classifier = classifier_optim.compute_gradients(loss_classifier, var_list=classifier_tvars, colocate_gradients_with_ops=True)
            classifier_train = classifier_optim.apply_gradients(grads_and_vars_classifier)    

    with tf.device("/gpu:0"):
        with tf.name_scope("discriminator_train"):
            discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
            discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            grads_and_vars_discrim = discrim_optim.compute_gradients(loss_discrim, var_list=discrim_tvars, colocate_gradients_with_ops=True)
            discrim_train = discrim_optim.apply_gradients(grads_and_vars_discrim)
         
    with tf.device("/gpu:0"):
        with tf.name_scope("generator_train"):
            with tf.control_dependencies([discrim_train]):
                gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
                gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
                train_generaotor_losses = loss_gen_reconstruction + loss_gen_classifer + loss_gen_discriminator
                grads_and_vars_gen = gen_optim.compute_gradients(train_generaotor_losses, var_list=gen_tvars, colocate_gradients_with_ops=True)
                gen_train = gen_optim.apply_gradients(grads_and_vars_gen)
                
    with tf.device("/gpu:1"):
        with tf.name_scope("attribute_network_train"):
            attribute_tvars = [var for var in tf.trainable_variables() if var.name.startswith("attribute_network")]
            attribute_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            grads_and_vars_attribute = attribute_optim.compute_gradients(loss_gen_GAN, var_list=discrim_tvars, colocate_gradients_with_ops=True)
            attribute_train = attribute_optim.apply_gradients(grads_and_vars_attribute)

        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        update_losses = ema.apply([loss_identity, 
                                   loss_classifier, 
                                   loss_discrim,  
                                   train_generaotor_losses,
                                   loss_gen_GAN])

        global_step = tf.train.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        identity_inputs = identity_inputs,
        attribute_inputs = attribute_inputs,
        predict_real=predict_real,
        predict_fake=predict_fake,
        loss_identity=ema.average(loss_identity),
        loss_classifier=ema.average(loss_classifier),
        loss_discrim=ema.average(loss_discrim),
        loss_gen_reconstruction=ema.average(loss_gen_reconstruction),
        loss_gen_classifer=ema.average(loss_gen_classifer),
        loss_gen_discriminator=ema.average(loss_gen_discriminator),
        loss_gen=ema.average(train_generaotor_losses),
        loss_gen_GAN=ema.average(loss_gen_GAN),
        accuracy_identity=accuracy_identity,
        accuracy_classifier=accuracy_classifier,
        grads_and_vars_identity=grads_and_vars_identity,
        grads_and_vars_classifier=grads_and_vars_classifier,
        grads_and_vars_discrim=grads_and_vars_discrim,
        grads_and_vars_gen=grads_and_vars_gen,
        grads_and_vars_attribute=grads_and_vars_attribute,
        outputs=gen_outputs,
        train=tf.group(update_losses, incr_global_step, identity_train, classifier_train, discrim_train, gen_train, attribute_train),
        onehot_identities=onehot_identities,
        probability_c_xs=logits_I,
        probability_c_outputs=logits_C
    )

##################### Main ###################################################
def main():
    ################# random seed ############################################
    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)
    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    
    ################ check directory #########################################
    if not os.path.exists(a.output_dir):
        #shutil.rmtree(a.output_dir)
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

    ############### display/save setting #####################################
    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    ################ read TFRecord dataset ###################################
    identity_batch, iterator  = read_tfrecord()
    attribute_batch, iterator  = read_tfrecord()
    
    ################ creat model  ############################################
    step_flag = tf.placeholder(tf.int32, shape=(), name="step_flag")
    # inputs and targets are of size [batch_size, height, width, channels]
    model = create_model(identity_batch.images, attribute_batch.images, identity_batch.labels, step_flag, class_number = a.num_identity)

    ################ summaries ###############################################
    # undo colorization splitting on images that we use for display/output
    identity_images = deprocess(identity_batch.images)
    attribute_images = deprocess(attribute_batch.images)
    identity_inputs = deprocess(model.identity_inputs)
    attribute_inputs = deprocess(model.attribute_inputs)
    outputs = deprocess(model.outputs)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_identity_images"):
        converted_identity_images = convert(identity_images)
    with tf.name_scope("convert_attribute_images"):
        converted_attribute_images = convert(attribute_images)
    with tf.name_scope("convert_identity_inputs"):
        converted_identity_inputs = convert(identity_inputs)
    with tf.name_scope("convert_attribute_inputs"):
        converted_attribute_inputs = convert(attribute_inputs)
    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)
    
    # YuhangLi: only save a part of images for saving driver space.
    num_display_images = 3000
    with tf.name_scope("encode_images"):
        display_fetches = {
            "identity_inputs": tf.map_fn(tf.image.encode_png, converted_identity_inputs[:num_display_images], dtype=tf.string, name="identity_inputs"),
            "attribute_inputs": tf.map_fn(tf.image.encode_png, converted_attribute_inputs[:num_display_images], dtype=tf.string, name="attribute_inputs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs[:num_display_images], dtype=tf.string, name="output_pngs")
        }

    with tf.name_scope("identity_summary"):
        tf.summary.image("identity_images", converted_identity_images)
    with tf.name_scope("attribute_summary"):
        tf.summary.image("attribute_images", converted_attribute_images)
    with tf.name_scope("identity_inputs_summary"):
        tf.summary.image("identity_inputs", converted_identity_inputs)
    with tf.name_scope("attribute_inputs_summary"):
        tf.summary.image("attribute_inputs", converted_attribute_inputs)
    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)            
    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))
    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("loss_identity", model.loss_identity)
    tf.summary.scalar("loss_classifier", model.loss_classifier)
    tf.summary.scalar("loss_discrim", model.loss_discrim)
    #tf.summary.scalar("loss_gen_reconstruction", model.loss_gen_reconstruction)
    #tf.summary.scalar("loss_gen_classifer", model.loss_gen_classifer)
    #tf.summary.scalar("loss_gen_discriminator", model.loss_gen_discriminator)
    tf.summary.scalar("loss_gen", model.loss_gen)
    tf.summary.scalar("loss_gen_GAN", model.loss_gen_GAN)
    tf.summary.scalar("accuracy_identity", model.accuracy_identity[1])
    tf.summary.scalar("accuracy_classifier", model.accuracy_classifier[1])

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    
    ################ configuration ##############################################
    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    ############### session ######################################################
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

        ######################## test ############################
        if a.mode == "test":
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
        
        ######################## train ############################
        else:
            """ Training
            Basic idea of training is:
                if should run something in this iteration:
                    add it to the fetches
                    .
                    .
                    .
                sess.run(fetches)
                
                if should run something in this iteration:
                    save the result of it
                    .
                    .
                    .
            "Something" includes: metadata, losses, summary, display(save) images
            """
            start = time.time()

            for step in range(max_steps):
                options = None
                run_metadata = None
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                ################### add to fetches #################################
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                     
                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):
                    fetches["loss_identity"] = model.loss_identity
                    fetches["loss_classifier"] = model.loss_classifier
                    fetches["loss_discrim"] = model.loss_discrim
                    #fetches["loss_gen_reconstruction"] = model.loss_gen_reconstruction
                    #fetches["loss_gen_classifer"] = model.loss_gen_classifer
                    #fetches["loss_gen_discriminator"] = model.loss_gen_discriminator
                    fetches["loss_gen"] = model.loss_gen
                    fetches["loss_gen_GAN"] = model.loss_gen_GAN
                    fetches["onehot_identities"] = model.onehot_identities
                    fetches["probability_c_xs"] = model.probability_c_xs
                    fetches["probability_c_outputs"] = model.probability_c_outputs


                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches
                
                ################ run session #########################################
                feed_dict = {step_flag: step}
                results = sess.run(fetches, feed_dict=feed_dict, options=options, run_metadata=run_metadata)

                ################ save/display results ################################
                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if 0:#should(a.display_freq):
                    print("saving display images")
                    filesets = save_images(results["display"], step=results["global_step"])
                    append_index(filesets, step=True)

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / identity_batch.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % identity_batch.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                    print("loss_discrim", results["loss_discrim"])
                    print("loss_identity", results["loss_identity"])
                    print("loss_classifier", results["loss_classifier"])
                    #print("loss_gen_reconstruction", results["loss_gen_reconstruction"])
                    #print("loss_gen_classifer", results["loss_gen_classifer"])
                    #print("loss_gen_discriminator", results["loss_gen_discriminator"])
                    print("loss_gen", results["loss_gen"])
                    print("loss_gen_GAN", results["loss_gen_GAN"])
                    
                    #print(results["onehot_identities"],results["onehot_identities"].shape)
                    #print(results["probability_c_xs"],results["probability_c_xs"].shape)
                    #print(results["probability_c_outputs"],results["probability_c_outputs"].shape)
                    #np.savetxt("onehot_identities.txt", results["onehot_identities"], fmt='%.5f')
                    #np.savetxt("probability_c_xs.txt", results["probability_c_xs"], fmt='%.5f')
                    #np.savetxt("probability_c_outputs.txt", results["probability_c_outputs"], fmt='%.5f')

                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

                if should(a.evaluate_freq):
                    print("evaluating results")
                    ## TODO
                    
                if sv.should_stop():
                    break
    return

if 1:
    main()