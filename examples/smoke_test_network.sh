#!/bin/bash

echo "Running network simulation & CNN training pipeline"


# echo "Step1 - Simulation"
# python scripts/03_simulate_static_network.py \
#   --dataFolderName data/network_smoke \
#   --segFolderName static_network_1 \
#   --simulationT 600000 \
#   --num_Xe 10 \
#   --num_Xi 10 \
#   --num_Se 80 \
#   --num_Si 20 \

# python scripts/03_simulate_static_network.py \
#   --dataFolderName data/network_smoke \
#   --segFolderName static_network_2 \
#   --simulationT 600000 \
#   --num_Xe 10 \
#   --num_Xi 10 \
#   --num_Se 60 \
#   --num_Si 40 \


# echo "Step2 - Generate DLs"
# # generate TTV dataloaders for CNN training
# python scripts/04_make_network_dataloaders.py \
#   --dataFolderName data/network_smoke \
#   --segFolderNamePrefix static_network \
#   --networkId_List 1

# generate one dataloaders for trained CNN to test on
python scripts/05_make_one_dataloader.py \
  --dataFolderName data/network_smoke \
  --segFolderName static_network_2 \
  --networkIdx 2


echo "Step3 - Training"
python scripts/06_train_connectivity_cnn.py \
  --dataFolderName data/network_smoke \
  --dataloaderType Raw \
  --dataloaderName TTV_1.pkl \
  --resultFolderName network_smoke \
  --unseen_dlName oneDL_2.pkl \
  --numRuns 10


echo "Smoke test finished."