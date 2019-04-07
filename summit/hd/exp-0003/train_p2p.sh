#!/bin/bash
cd /gdata/liyh/project/e2p

python p2p_st.py \
--mode train \
--output_dir /gdata/liyh/data/CelebA-HD/checkpoint/exp-0003_p2p \
--max_epochs 2000 \
--input_dir /gdata/liyh/data/CelebA-HD/data/tfrecord  \
--batch_size 8 \
--num_examples 24000 \
--generator ed \
--discriminator conv \
--input_type edge \
--scale_size 256 \
--target_size 256 \
--num_gpus 2 \
--stabilization wgan \
--no_double_D \
--no_fm \
--no_style_loss