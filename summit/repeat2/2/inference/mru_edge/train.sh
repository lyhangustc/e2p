#!/bin/bash
cd /gdata/liyh/project/p2p

err_f=/gdata/liyh/data/CelebA/checkpoint/mru_edge/inference_err_file.err
log_f=/gdata/liyh/data/CelebA/checkpoint/mru_edge/inference_out_file.out

python p2p_mru.py \
--mode test \
--output_dir /gdata/liyh/data/CelebA/output/mru_edge \
--checkpoint /gdata/liyh/data/CelebA/checkpoint/mru_edge \
--input_dir /gdata/liyh/data/CelebA/data/test  \
--batch_size 8 \
--num_examples 60758 \
--generator mru \
--input_type edge