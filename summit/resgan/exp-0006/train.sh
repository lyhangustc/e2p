#!/bin/bash
cd /gdata/liyh/project/e2p

python p2p_att.py \
--mode train \
--output_dir /gdata/liyh/data/CelebA/checkpoint/resgan_df/exp-0005_restore \
--checkpoint /gdata/liyh/data/CelebA/checkpoint/resgan_df/exp-0005-1 \
--max_epochs 2000 \
--input_dir /gdata/liyh/data/CelebA/data/train  \
--batch_size 8 \
--num_examples 141633 \
--generator resgan \
--discriminator resgan \
--input_type df