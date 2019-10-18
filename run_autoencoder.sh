#!/bin/bash
#gsutil cp gs://bepro-server-storage/dsort_tracking_data/"$rec_id"/hypotheses_1.txt dsort_out/ 
source activate open-mmlab
gsutil cp gs://bepro-server-storage/dsort_tracking_data/autoencoder_data/pwd_dataset.txt ./08-AutoEncoder/
python ./08-AutoEncoder/bepro_conv_autoencoder.py ./08-AutoEncoder/pwd_dataset.txt