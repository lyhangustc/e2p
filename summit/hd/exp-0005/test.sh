#!/bin/bash
cd /gdata/liyh/project/e2p


for GAMMA in 0.0 0.3 0.6
do 
    python p2p_st.py \
    --input_dir /gdata/liyh/data/CelebA-HD/data/tfrecord/test \
    --output_dir /gdata/liyh/data/CelebA-HD/output/vary_gamma/exp-0003_p2p/$GAMMA \
    --checkpoint /gdata/liyh/data/CelebA-HD/checkpoint/exp-0003_p2p_df \
    --mode test \
    --input_type vg \
    --df_norm value \
    --num_example 6000 \
    --generator ed \
    --discriminator conv \
    --scale_size 256 \
    --target_size 256 \
    --stabilization wgan \
    --no_fm \
    --no_style_loss \
    --no_double_D \
    --df_threshold $GAMMA
done