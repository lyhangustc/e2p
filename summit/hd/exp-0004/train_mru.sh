#!/bin/bash
cd /gdata/liyh/project/e2p

python p2p_st.py \
--input_dir /gdata/liyh/data/CelebA-HD/data/tfrecord \
--output_dir /gdata/liyh/data/CelebA-HD/checkpoint/exp-0004_finetune_sa_mru \
--checkpoint /gdata/liyh/data/CelebA-HD/checkpoint/exp-0003_mru_NoFM_NoS_NoG \
--mode train \
--input_type df \
--df_norm value \
--num_example 24000 \
--generator mru \
--discriminator resgan \
--scale_size 256 \
--target_size 256 \
--stabilization wgan \
--no_fm \
--no_style_loss \
--finetune \
--batch_size 2 \
--no_double_D
