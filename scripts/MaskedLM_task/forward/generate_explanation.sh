#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --account=def-emilios
#SBATCH --mem=512G 
#SBATCH --mail-type=ALL # 发送哪一种email通知：BEGIN,END,FAIL,ALL
#SBATCH --mail-user=xm863858@dal.ca # 把通知发送到哪一个邮箱

scriptDir=scripts
model=../../distilbert-base-cased
inputPath=../../../data # path to a sentence file

saveDir=.

python ${scriptDir}/generate_explanation.py --dataset-name-or-path ${inputPath}/sentences_MASK.txt --model-name ${model} --tokenizer-name ${model} --save-dir ${saveDir}

