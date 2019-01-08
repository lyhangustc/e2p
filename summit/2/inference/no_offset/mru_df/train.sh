#!/bin/bash
cd /gdata/liyh/project/p2p

err_f=/gdata/liyh/data/CelebA/checkpoint/no_offset/mru_df_s/err_file1.err
log_f=/gdata/liyh/data/CelebA/checkpoint/no_offset/mru_df_s/out_file1.out

python p2p_mru.py \
--mode train \
--output_dir /gdata/liyh/data/CelebA/checkpoint/no_offset/mru_df_s \
--max_epochs 2000 \
--input_dir /gdata/liyh/data/CelebA/data/train  \
--batch_size 8 \
--num_examples 141633 \
--generator mru \
--input_type df