from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import glob
from PIL import Image
import tensorflow as tf
import scipy.io as sio
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--A_dir", help="path to folder containing A images")
parser.add_argument("--B_dir", help="path to folder containing B files")
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--save_name", type=str, default="data", help="save file name")
# parser.add_argument("--label", required=True, help="label of mat")
a = parser.parse_args()

crop_size = 286

def crop_center(img):
    width = img.size[0]
    height = img.size[1]
    size = min(width, height)
    offset_width = (width - size) // 2
    offset_height = (height - size) // 2
    img = img.crop((offset_width, offset_height, size, size))
    return img

def images2npy():
    if a.A_dir is None or not os.path.exists(a.A_dir):
        raise Exception("A_dir does not exist")
    if a.B_dir is None or not os.path.exists(a.B_dir):
        raise Exception("B_dir does not exist")    
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)
    
    A_paths = glob.glob(os.path.join(a.A_dir, "*.jpg"))
    if len(A_paths) == 0:
        A_paths = glob.glob(os.path.join(a.A_dir, "*.png"))
    if len(A_paths) == 0:
        raise Exception("A_dir contains no image files")
    
    A_list = []
    B_list = []
    
    count = 0
    
    for i, A_path in enumerate(A_paths):
        print(i)
        basename, _ = os.path.splitext(os.path.basename(A_path))
        
        if basename[0]=='f':
            basename = 'F2'+ basename[1:] + '-sz1'
        if basename[0]=='m':
            basename = 'M2'+ basename[1:] + '-sz1'
        
        B_path = os.path.join(a.B_dir, basename + '.jpg')
        
        A_image = Image.open(A_path)
        A_image = crop_center(A_image)
        A_image = A_image.resize((crop_size,crop_size))
        A_array = np.asarray(A_image, dtype=np.uint8)        
        B_image = Image.open(B_path)
        B_image = crop_center(B_image)
        B_image = B_image.resize((crop_size,crop_size))
        B_array = np.asarray(B_image, dtype=np.uint8)   
        
        print(B_array.shape)
        print(A_array.shape)     
                
        assert(B_array.shape[0] == A_array.shape[0] and \
            B_array.shape[1] == A_array.shape[1])

        A_list.append(A_array)
        B_list.append(B_array)
        count = count + 1
    A_filename = os.path.join(a.output_dir, a.save_name + '_A.npy')
    B_filename = os.path.join(a.output_dir, a.save_name + '_B.npy')
    A_npy = np.asarray(A_list)
    B_npy = np.asarray(B_list)
    np.save(A_filename, A_npy)
    np.save(B_filename, B_npy)

images2npy()