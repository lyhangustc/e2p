#!/bin/bash
cd /gdata/liyh/project/e2p

python p2p_st.py \
--mode train \
--output_dir /gdata/liyh/data/CelebA-HD/output/exp-0004_finetune_sa_last1 \
--input_dir /gdata/liyh/data/CelebA-HD/data/tfrecord  \
--batch_size 1 \
--num_examples 3000 \
--generator resgan \
--discriminator resgan \
--input_type df \
--scale_size 256 \
--target_size 256 \
--num_gpus 2  \
--checkpoint /gdata/liyh/data/CelebA-HD/checkpoint/exp-0001_256_D \
--finetune