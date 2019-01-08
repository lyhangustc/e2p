#!/bin/bash
cd /gdata/liyh/project/p2p

python p2p_att.py \
--mode train \
--output_dir /gdata/liyh/data/CelebA/atten/checkpoint/exp-0000 \
--max_epochs 2000 \
--input_dir /gdata/liyh/data/CelebA/data/train  \
--batch_size 16 \
--num_examples 141633 \
--generator sa \
--input_type df \
--enc_atten FFFFF \
--dec_atten FFFFF