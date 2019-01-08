#!/bin/bash
cd /gdata/liyh/project/p2p
output_file = "/gdata/liyh/project/p2p/summit/eval/results"

python /gdata/liyh/tool/InceptionScore/ComputeInceptionScore.py --input_dir /gdata/liyh/data/CelebA/old/output/mru_df/images --key_word output >> /gdata/liyh/project/p2p/summit/eval/results1
python /gdata/liyh/tool/InceptionScore/ComputeInceptionScore.py --input_dir /gdata/liyh/data/CelebA/old/output/ed_edge/images --key_word output >> /gdata/liyh/project/p2p/summit/eval/results2
python /gdata/liyh/tool/InceptionScore/ComputeInceptionScore.py --input_dir /gdata/liyh/data/CelebA/old/output/ed_df/images --key_word output >> /gdata/liyh/project/p2p/summit/eval/results3
python /gdata/liyh/tool/InceptionScore/ComputeInceptionScore.py --input_dir /gdata/liyh/data/CelebA/old/output/mru_edge/images --key_word output >> /gdata/liyh/project/p2p/summit/eval/results4
