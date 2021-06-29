#This is SH file for LearningAIM

date

export PYTHONPATH=.:$PYTHONPATH
# random seed for train/val split
seed_number=42
# dataset options <'HRF-AV', 'LES-AV', 'DRIVE_AV'>
dataset_name='LES-AV'

CUDA_VISIBLE_DEVICES=0 python train.py --e=1500 \
                                        --batch-size=2 \
                                        --learning-rate=8e-4 \
                                        --v=10.0 \
                                        --alpha=0.5 \
                                        --beta=1.1 \
                                        --gama=0.08 \
                                        --dataset=${dataset_name} \
                                        --discriminator='unet' \
                                        --job_name=20210628_${dataset_name}_randomseed_${seed_number} \
                                        --uniform=False \
                                        --seed_num=${seed_number}

date


