#!/bin/bash
cd /gdata/liyh/project/e2p

python p2p_HD.py \
--mode train \
--output_dir /gdata/liyh/data/CelebA-HD/checkpoint/exp-0001_256_MT_E_NoStyle_lr10x \
--max_epochs 2000 \
--input_dir /gdata/liyh/data/CelebA-HD/data/tfrecord  \
--batch_size 8 \
--num_examples 27000 \
--generator resgan \
--discriminator resgan \
--input_type df \
--target_size 256 \
--lr_gen 0.002 \
--lr_discrim 0.0002 \
--num_gpus 8 \
--no_style_loss

