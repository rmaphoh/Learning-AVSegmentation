#This is SH file for LearningAIM, testing on customise data
date

export PYTHONPATH=.:$PYTHONPATH

seed_number=42
dataset_name='ALL-AV'
test_checkpoint=1401

CUDA_VISIBLE_DEVICES=0 python test_customise.py --batch-size=1 \
                                                --dataset=${dataset_name} \
                                                --job_name=20210628_${dataset_name}_randomseed_${seed_number} \
                                                --checkstart=${test_checkpoint} \
                                                --uniform=True \
                                                --customise_data='/content/drive/MyDrive/customise_image/' \
                                                --customise_output='/content/drive/MyDrive/customise_output/'

date




