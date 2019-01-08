#!/bin/bash
cd /gdata/liyh/project/p2p

err_f=/gdata/liyh/data/CelebA/checkpoint/ed_edge/err_file1.err
log_f=/gdata/liyh/data/CelebA/checkpoint/ed_edge/out_file1.out

python p2p_mru.py \
--mode train \
--output_dir /gdata/liyh/data/CelebA/checkpoint/ed_edge \
--max_epochs 2000 \
--input_dir /gdata/liyh/data/CelebA/data/train  \
--batch_size 8 \
--num_examples 141633 \
--generator ed \
--input_type edge