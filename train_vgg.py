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
from vgg import *

import warnings
from functools import partial
from nets import resnet_utils
from models import *
import ops

slim = tf.contrib.slim

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mode", default="train", choices=["train", "test", "export"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--class_num", type=int, required=True)
parser.add_argument("--num_examples", type=int, required=True)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--batch_size", type=int, default=8, help="number of images in batch")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--target_size", type=int, default=224, help="scale images to this size before cropping to 256x256")

parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=1000, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")
parser.add_argument("--evaluate_freq", type=int, default=5000, help="evaluate training data every save_freq steps, 0 to disable")

a = parser.parse_args()

EPS = 1e-12

Examples = collections.namedtuple("Examples", "images, labels, steps_per_epoch")
Model = collections.namedtuple("Model", "logits,onehot_labels,accuracy, label_indices, loss, grads_and_vars, train")

def convert(image):
    return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)



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
    label = parsed_features['label']
    image = tf.reshape(image, [height, width, 3])
    image = tf.image.convert_image_dtype(image, dtype=tf.float64)
    image = image * 2. -1.    

    seed = random.randint(0, 2**31 - 1)
    image = transform(image, seed) 

    return image, label

def parse_function_cifar10(example_proto):
    '''

    '''            
    features = {
                'image/encoded': tf.FixedLenFeature([], tf.string),
                'image/format': tf.FixedLenFeature([], tf.string),
                'image/class/label': tf.FixedLenFeature([], tf.int64),
                'image/height': tf.FixedLenFeature([], tf.int64),
                'image/width': tf.FixedLenFeature([], tf.int64),
            }        
    
    parsed_features = tf.parse_single_example(example_proto, features=features) 
    
    #image = tf.decode_raw(parsed_features['image/encoded'], tf.uint8)
    image = tf.image.decode_png(parsed_features['image/encoded'], tf.uint8)
    height = tf.cast(parsed_features['image/height'], tf.int32)
    width = tf.cast(parsed_features['image/width'], tf.int32)
    label = parsed_features['image/class/label']
    image = tf.reshape(image, [height, width, 4])
    image = tf.image.convert_image_dtype(image[:,:,:3], dtype=tf.float32)
    image = image * 2. -1.    

    seed = random.randint(0, 2**31 - 1)
    image = transform(image, seed) 

    return image, label

def read_tfrecord():
    tfrecord_fn = glob.glob(os.path.join(a.input_dir, "*.tfrecords"))
    dataset = tf.data.TFRecordDataset(tfrecord_fn)
    #dataset = dataset.map(parse_function)  # Parse the record into tensors. 
    dataset = dataset.map(parse_function_cifar10)  # Parse the record into tensors. 
    dataset = dataset.repeat()  # Repeat the input indefinitely.
    #dataset = dataset.shuffle(buffer_size=10000)
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
    

def create_model(images, label_indices, class_num):
    onehot_labels = tf.one_hot(label_indices, class_num)

    with tf.name_scope("vgg_network"):
        logits, endpoints = vgg_16(images, num_classes=class_num,global_pool=True)
    with tf.name_scope("softmax_cross_entropy_loss"):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    with tf.name_scope("accuracy"):
        accuracy = tf.metrics.accuracy(labels=label_indices, predictions=tf.argmax(logits, axis=1))
    
    with tf.name_scope("training_network"):
        tvars = [var for var in tf.trainable_variables()]
        optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        grads_and_vars = optim.compute_gradients(loss, var_list=tvars, colocate_gradients_with_ops=True)
        train_op = optim.apply_gradients(grads_and_vars)

        #with tf.control_dependencies([train_op]):
        #    ema = tf.train.ExponentialMovingAverage(decay=0.99)
        #    update_losses = ema.apply([loss])
        global_step = tf.train.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        logits=logits,
        onehot_labels=onehot_labels,
        accuracy=accuracy,
        label_indices=label_indices,
        #loss=ema.average(loss),
        loss=loss,
        grads_and_vars=grads_and_vars,
        #train=tf.group(update_losses, incr_global_step),
        train=tf.group(train_op, incr_global_step),
        )

def main():
    ################# random seed ############################################
    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)
    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    
    ################ check directory #########################################
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

    ############### display/save setting #####################################
    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    ################ read TFRecord dataset ###################################
    input_batch, iterator  = read_tfrecord()

    ################ creat model  ############################################
    # inputs and targets are of size [batch_size, height, width, channels]
    model = create_model(input_batch.images, input_batch.labels, a.class_num)

    ################ summaries ###############################################
    # undo colorization splitting on images that we use for display/output
    images = deprocess(input_batch.images)

    # YuhangLi: only save a part of images for saving driver space.
    with tf.name_scope("encode_images"):
        display_fetches = {
            #"paths": input_batch.paths,
            "images": tf.map_fn(tf.image.encode_png, convert(images), dtype=tf.string, name="images"),
            "label": input_batch.labels
        }

    with tf.name_scope("images_summary"):
        tf.summary.image("images", convert(images))
    #with tf.name_scope("label"):
    #    tf.summary.image("label", input_batch.labels)            

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.accuracy[1])

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)
        #tf.summary.histogram(var.op.name + "/gradients", grad)
    
    ################ configuration ##############################################
    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    f = open("log.txt",mode='wb')

    ############### session ######################################################
    with sv.managed_session(config=sess_config) as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = input_batch.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        ######################## test ############################
        if a.mode == "test":
            # at most, process the test data once
            max_steps = min(input_batch.steps_per_epoch, max_steps)
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
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)
                options = None
                run_metadata = None

                ################### add to fetches #################################
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                    "label_indices":model.label_indices,
                    "onehot_labels":model.onehot_labels,
                    "logits":model.logits,
                }

                if should(a.progress_freq):
                    fetches["loss"] = model.loss
                    fetches["accuracy"] = model.accuracy

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                
                results = sess.run(fetches, options=options, run_metadata=run_metadata)
                #print(results["label_indices"])
                ################ save/display results ################################
                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / input_batch.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % input_batch.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                    print("loss", results["loss"])
                    print("accuracy", results["accuracy"])
                    print("label_indices", results["label_indices"])
                    
                    np.savetxt("logits", results["logits"], fmt='%.4f')
                    np.savetxt("onehot_labels", results["onehot_labels"], fmt='%.4f')

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