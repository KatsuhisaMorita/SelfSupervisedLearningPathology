#!/bin/sh
# sample code for training
python3 src/SelfSupervisedLearningPathology/tggate/train/train_tggate_ext.py \
--note 'barlowtwins_fold0' --model_name ResNet18 \
--ssl_name 'barlowtwins' \
--seed 0 --fold 0 --dir_result 'fold/bt0' \
--resume_epoch 51 #--resume