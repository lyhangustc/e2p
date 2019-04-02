#!/bin/bash
cd /gdata/liyh/project/e2p

python p2p_st.py \
--mode train \
--output_dir /gdata/liyh/data/CelebA-HD/checkpoint/exp-0003_mru_NoFM_NoS_NoG \
--input_dir /gdata/liyh/data/CelebA-HD/data/tfrecord  \
--batch_size 8 \
--num_examples 27000 \
--generator mru\
--discriminator resgan \
--input_type df \
--scale_size 256 \
--target_size 256 \
--num_gpus 2  \
--no_fm \
--no_style \
--no_double_D