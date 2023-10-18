#!/bin/bash

#python process_gender_related_concept.py --masked_info_file ../masked_representation_6.csv --masked_concept_file ../predictions_layer_6.csv --data_file ../sentence_MASK.txt --output_file concept_sentence.csv

python3.8 visualize_concept_with_sentence.py --concept_sentence_file concept_sentence.csv --concept_cluster_file clusters-300.txt --output_file visualization_layer6.pdf