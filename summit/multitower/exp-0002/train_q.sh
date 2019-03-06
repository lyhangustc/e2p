#!/bin/bash
cd /gdata/liyh/project/e2p

python p2p_att.py \
--mode train \
--output_dir /gdata/liyh/data/CelebA/checkpoint/multitower/exp-0002_q \
--max_epochs 2000 \
--input_dir /gdata/liyh/data/CelebA/data/train  \
--batch_size 8 \
--num_examples 141633 \
--generator resgan \
--discriminator resgan \
--input_type df \
--num_gpus 4 \
--num_gpus_per_tower 2
