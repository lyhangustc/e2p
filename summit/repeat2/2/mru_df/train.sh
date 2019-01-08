#!/bin/bash
cd /gdata/liyh/project/p2p

python p2p_att.py \
--mode train \
--output_dir /gdata/liyh/data/CelebA/new/checkpoint/mru_df_1 \
--max_epochs 2000 \
--input_dir /gdata/liyh/data/CelebA/data/train  \
--batch_size 8 \
--num_examples 141633 \
--generator mru \
--input_type df