#!/usr/bin/env bash

# load modules
module load miniconda

# set variables
export OMP_NUM_THREADS=4
thred=32

# run ML sets
parallel -j $thred python svm_knn.py \
  -i drug_input.csv \
  -o svm_knn{}.txt \
  ::: $(seq 0 999)

parallel -j $thred python svm_rfc.py \
  -i drug_input.csv \
  -o svm_rfc{}.txt \
  ::: $(seq 0 999)

parallel -j $thred python xgb_knn.py \
  -i drug_input.csv \
  -o xgb_knn{}.txt \
  ::: $(seq 0 999)

parallel -j $thred python xgb_rfc.py \
  -i drug_input.csv \
  -o xgb_rfc{}.txt \
  ::: $(seq 0 999)

# generate outfiles
python ML.outcomes.py
