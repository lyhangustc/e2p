#!/bin/bash
cd /gdata/liyh/project/e2p

python p2p_HD.py \
--mode train \
--output_dir /gdata/liyh/data/CelebA-HD/checkpoint/exp-0003_256_D \
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
--stabilization wgan  