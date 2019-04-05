#!/bin/bash
cd /gdata/liyh/project/e2p

for GAMMA in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 
do   
    python p2p_st.py \
    --input_dir /gdata/liyh/data/CelebA-HD/data/tfrecord/test \
    --output_dir /gdata/liyh/data/CelebA-HD/output/vary_gamma/exp-0004_finetune_sa_last2_end_dgx/$GAMMA \
    --checkpoint /gdata/liyh/data/CelebA-HD/checkpoint/exp-0004_finetune_sa_last2_end_dgx \
    --mode test \
    --input_type vg \
    --df_norm value \
    --num_example 6000 \
    --generator mru_res \
    --discriminator conv \
    --scale_size 256 \
    --target_size 256 \
    --stabilization wgan \
    --batch_size 2 \
    --no_double_D \
    --use_attention \
    --df_threshold $GAMMA
done



