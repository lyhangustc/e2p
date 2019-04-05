#!/bin/bash
cd /gdata/liyh/project/e2p

python p2p_st.py \
--input_dir /gdata/liyh/data/CelebA-HD/data/tfrecord \
--output_dir /gdata/liyh/data/CelebA-HD/checkpoint/exp-0004_finetune_sa_mru_res_FM_S \
--checkpoint /gdata/liyh/data/CelebA-HD/checkpoint/exp-0003_mru_res1_NoFM_NoS_NoG_convD \
--mode train \
--input_type df \
--df_norm value \
--num_example 24000 \
--generator mru_res \
--discriminator conv \
--scale_size 256 \
--target_size 256 \
--stabilization wgan \
--finetune \
--batch_size 2 \
--no_double_D