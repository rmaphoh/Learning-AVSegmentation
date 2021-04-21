#This is SH file for LearningAIM

source activate torch

date

export PYTHONPATH=.:$PYTHONPATH

seed_number=42
dataset_name='HRF-AV'
test_checkpoint=11

CUDA_VISIBLE_DEVICES=0 python test.py --batch-size=1 \
                                        --dataset=${dataset_name} \
                                        --job_name=20210420_${dataset_name}_randomseed_${seed_number} \
                                        --checkstart=${test_checkpoint} \
                                        --uniform=False

date



