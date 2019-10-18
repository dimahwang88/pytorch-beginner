#!/bin/bash
#gsutil cp gs://bepro-server-storage/dsort_tracking_data/"$rec_id"/hypotheses_1.txt dsort_out/ 
#gsutil cp gs://bepro-server-storage/dsort_tracking_data/autoencoder_data/association_dataset.zip .
#unzip association_dataset.zip

cd ./training
find "$PWD" >> pwd_dataset.txt
cd ../

source activate open-mmlab
python ./08-AutoEncoder/bepro_conv_autoencoder.py ./training/pwd_dataset.txt