#!/usr/bin/env bash

for filename in /mnt/nfs/scratch1/ahandler/experiments/qsr/*json
do 
    sleep .25
    sbatch -p titanx-long --gres=gpu:1 nn/go.sh $filename --mem=12GB
done
