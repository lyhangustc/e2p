#!/bin/bash
cd /gdata/liyh/project/p2p

python p2p_mru.py \
--mode train \
--output_dir /gdata/liyh/data/CelebA/atten/checkpoint/sa_df \
--max_epochs 2000 \
--input_dir /gdata/liyh/data/CelebA/data/train  \
--batch_size 1 \
--num_examples 141633 \
--generator sa \
--input_type df