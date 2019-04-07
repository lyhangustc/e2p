#!/bin/bash
cd /gdata/liyh/project/e2p

for GAMMA in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 
do 
    python /gdata/liyh/tool/InceptionScore/ComputeInceptionScore.py \
    --input_dir /gdata/liyh/data/CelebA-HD/output/vary_gamma/exp-0004_finetune_sa_mru_res_FM_S/$GAMMA/images \
    --real_dir /gdata/liyh/data/CelebA-HD/data/image/test \
    --score_type IS \
    --num_input 1000 \
    --num_real 1000 \
    --batch_size 200
    python /gdata/liyh/tool/InceptionScore/ComputeInceptionScore.py \
    --input_dir /gdata/liyh/data/CelebA-HD/output/vary_gamma/exp-0004_finetune_sa_mru_res_FM_S/$GAMMA/images \
    --real_dir /gdata/liyh/data/CelebA-HD/data/image/test \
    --score_type FID \
    --num_input 1000 \
    --num_real 1000 \
    --batch_size 200
    python /gdata/liyh/tool/InceptionScore/ComputeInceptionScore.py \
    --input_dir /gdata/liyh/data/CelebA-HD/output/vary_gamma/exp-0004_finetune_sa_mru_res_FM_S/$GAMMA/images \
    --real_dir /gdata/liyh/data/CelebA-HD/data/image/test \
    --score_type KID \
    --num_input 1000 \
    --num_real 1000 \
    --batch_size 200

done

for GAMMA in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 
do 
    python /gdata/liyh/tool/InceptionScore/ComputeInceptionScore.py \
    --input_dir /gdata/liyh/data/CelebA-HD/output/vary_gamma/exp-0004_finetune_sa_last2_end_dgx/$GAMMA/images \
    --real_dir /gdata/liyh/data/CelebA-HD/data/image/test \
    --score_type IS \
    --num_input 1000 \
    --num_real 1000 \
    --batch_size 200
    python /gdata/liyh/tool/InceptionScore/ComputeInceptionScore.py \
    --input_dir /gdata/liyh/data/CelebA-HD/output/vary_gamma/exp-0004_finetune_sa_last2_end_dgx/$GAMMA/images \
    --real_dir /gdata/liyh/data/CelebA-HD/data/image/test \
    --score_type FID \
    --num_input 1000 \
    --num_real 1000 \
    --batch_size 200
    python /gdata/liyh/tool/InceptionScore/ComputeInceptionScore.py \
    --input_dir /gdata/liyh/data/CelebA-HD/output/vary_gamma/exp-0004_finetune_sa_last2_end_dgx/$GAMMA/images \
    --real_dir /gdata/liyh/data/CelebA-HD/data/image/test \
    --score_type KID \
    --num_input 1000 \
    --num_real 1000 \
    --batch_size 200

done
