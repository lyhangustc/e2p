#!/bin/bash
cd /gdata/liyh/project/p2p

err_f=/gdata/liyh/project/p2p/summit/atten/exp-0000/err_file1.err
log_f=/gdata/liyh/project/p2p/summit/atten/exp-0000/out_file1.out

python p2p_att.py \
--mode train \
--output_dir /gdata/liyh/data/CelebA/atten/checkpoint/exp-0000 \
--max_epochs 2000 \
--input_dir /gdata/liyh/data/CelebA/data/train  \
--batch_size 16 \
--num_examples 141633 \
--generator sa \
--input_type df