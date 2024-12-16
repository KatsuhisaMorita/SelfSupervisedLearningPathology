#!/bin/sh
python3 /workspace/pathology/src/SelfSupervisedLearningPathology/analyzer/visualizer.py \
--filein "/workspace/HDD1/TGGATEs/WSI/Liver/29_day/28214.svs" \
--dir_featurize_model "/workspace/pathology/models/fold/bt0/model_ssl.pt" \
--dir_classification_models="/workspace/pathology/models/analyzer/bt_layer45_pu.pickle" \
--rawimage --anomaly --anomaly_crops --findings --findings_crops --only_highscore \
--savedir /workspace/pathology/result/finding/visualizer_test/