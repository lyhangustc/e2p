#!/bin/bash
cd /gdata/liyh/project/e2p

python p2p_HD.py \
--mode train \
--output_dir /gdata/liyh/data/CelebA-HD/checkpoint/exp-0000_E \
--max_epochs 2000 \
--input_dir /gpub/temp/CelebA-HQ/tfrecord  \
--batch_size 4 \
--num_examples 27000 \
--generator resgan \
--discriminator resgan \
--input_type df \
--target_size 512  \
--num_gpus 8