#!/bin/bash
cd /gdata/liyh/project/e2p

for GAMMA in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 
do 
    python p2p_st.py \
    --input_dir /gdata/liyh/data/CelebA-HD/data/tfrecord/test \
    --output_dir /gdata/liyh/data/CelebA-HD/output/vary_gamma/test_exp_0001_256_D_vg/$GAMMA \
    --checkpoint /gdata/liyh/data/CelebA-HD/checkpoint/exp-0001_256_D \
    --mode test \
    --input_type vg \
    --df_norm value \
    --num_example 6000 \
    --generator resgan \
    --discriminator resgan \
    --scale_size 256 \
    --target_size 256 \
    --stabilization wgan \
    --no_fm \
    --no_style_loss \
    --df_threshold $GAMMA
done


