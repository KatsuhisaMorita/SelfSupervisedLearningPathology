#!/bin/sh
# sample code for featurize (all patches in WSI)
python src/SelfSupervisedLearningPathology/tggate/featurization/wsi_masked.py \
--batch_size 512 --model_name ResNet18 --ssl_name barlowtwins \
--model_path  model/bt0/model_ssl.pt \ #change to your dir
--dir_result feature/fold \ #change to your dir
--result_name fold0_ \
--seed 0  --tggate_all
