#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --account=def-emilios
#SBATCH --mem=256G
#SBATCH --mail-type=ALL # 发送哪一种email通知：BEGIN,END,FAIL,ALL
#SBATCH --mail-user=xm863858@dal.ca # 把通知发送到哪一个邮箱

scriptDir=scripts
inputPath=../../clustering/ # path to a sentence file
input=sentences.txt.tok.sent_len #name of the sentence file

mkdir "winoBias"

# maximum sentence length
sentence_length=300
# analyze latent concepts of layer 12


#1. Tokenize text with moses tokenizer
# perl ${scriptDir}/tokenizer/tokenizer.perl -l en -no-escape < ${inputPath}/$input > $input.tok

python ${scriptDir}/gender_convert_to_MASK.py --input_file ${inputPath}/$input --output_file ../../../data/sentences_MASK.txt

input=sentences_MASK.txt
#2. Do sentence length filtering and keep sentences max length of 300
python ${scriptDir}/sentence_length.py --text-file ../../../data/$input --length ${sentence_length} --output-file $input.tok.sent_len

working_file=$input.tok.sent_len #do not change this
#3. Modify the input file to be compatible with the model
python ${scriptDir}/modify_input.py --text-file $working_file --output-file $input.tok.sent_len.modified

#4. Calculate vocabulary size
python ${scriptDir}/frequency_count.py --input-file ${working_file}.modified --output-file ${working_file}.words_freq

