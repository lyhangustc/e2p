#!/bin/bash
cd /gdata/liyh/project/e2p

python p2p_HD.py \
--mode train \
--output_dir /gdata/liyh/data/CelebA-HD/checkpoint/exp-0002_weight_100_1_10_10 \
--max_epochs 2000 \
--input_dir /gdata/liyh/data/CelebA-HD/data/tfrecord  \
--batch_size 8 \
--num_examples 27000 \
--generator resgan \
--discriminator conv \
--input_type df \
--scale_size 256 \
--target_size 256 \
--num_gpus 2 \
--stabilization lsgan \
--l1_weight 100.0 \
--gan_weight 1.0 \
--fm_weight 10.0 \
--style_weight 10.0
