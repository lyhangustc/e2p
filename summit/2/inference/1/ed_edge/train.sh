#!/bin/bash
cd /gdata/liyh/project/p2p

err_f=/gdata/liyh/data/CelebA/checkpoint/1/ed_edge/err_file1.err
log_f=/gdata/liyh/data/CelebA/checkpoint/1/ed_edge/out_file1.out

python p2p_mru.py \
--mode test \
--output_dir /gdata/liyh/data/CelebA/output/1ed_edge \
--checkpoint /gdata/liyh/data/CelebA/checkpoint/1/ed_edge \
--input_dir /gdata/liyh/data/CelebA/data/test  \
--batch_size 8 \
--num_examples 60758 \
--generator ed \
--input_type edge