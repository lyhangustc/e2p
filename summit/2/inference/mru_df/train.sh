#!/bin/bash
cd /gdata/liyh/project/p2p

err_f=/gdata/liyh/data/CelebA/checkpoint/mru_df/err_file1.err
log_f=/gdata/liyh/data/CelebA/checkpoint/mru_df/out_file1.out

python p2p_mru.py \
--mode test \
--output_dir /gdata/liyh/data/CelebA/output/mru_df \
--checkpoint /gdata/liyh/data/CelebA/checkpoint/mru_df \
--input_dir /gdata/liyh/data/CelebA/data/test  \
--batch_size 8 \
--num_examples 60758 \
--generator mru \
--input_type df