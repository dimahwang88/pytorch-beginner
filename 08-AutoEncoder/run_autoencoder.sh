#!/bin/bash
#gsutil cp gs://bepro-server-storage/dsort_tracking_data/"$rec_id"/hypotheses_1.txt dsort_out/ 
gsutil cp gs://bepro-server-storage/dsort_tracking_data/autoencoder_data/pwd_dataset.txt .
python bepro_conv_autoencoder.py pwd_dataset.txt