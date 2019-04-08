#!/bin/bash
cd /gdata/liyh/project/e2p

GAMMA=0.0
for DIR in exp-0003_p2p exp-0003_mru exp-0004_finetune_sa_last2_end_dgx test_0003_mru_NoFM_NoS_NoG test_0003_mru_res1_NoFM_NoS_NoG_convD
do 
    python /gdata/liyh/tool/InceptionScore/ComputeInceptionScore.py \
    --input_dir /gdata/liyh/data/CelebA-HD/output/vary_gamma/$DIR/0.0/images \
    --real_dir /gdata/liyh/data/CelebA-HD/data/image/test \
    --score_type IS \
    --num_input 1000 \
    --num_real 1000 \
    --batch_size 200
    python /gdata/liyh/tool/InceptionScore/ComputeInceptionScore.py \
    --input_dir /gdata/liyh/data/CelebA-HD/output/vary_gamma/$DIR/0.0/images \
    --real_dir /gdata/liyh/data/CelebA-HD/data/image/test \
    --score_type FID \
    --num_input 1000 \
    --num_real 1000 \
    --batch_size 200
    python /gdata/liyh/tool/InceptionScore/ComputeInceptionScore.py \
    --input_dir /gdata/liyh/data/CelebA-HD/output/vary_gamma/$DIR/0.0/images \
    --real_dir /gdata/liyh/data/CelebA-HD/data/image/test \
    --score_type KID \
    --num_input 1000 \
    --num_real 1000 \
    --batch_size 200

done

