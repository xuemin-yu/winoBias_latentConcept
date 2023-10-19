#!/bin/bash

#python process_gender_related_concept.py --masked_info_file distilbert-base-cased/masked_representation_6.csv --masked_concept_file distilbert-base-cased/predictions_layer_6.csv --data_file sentence_MASK.txt --output_file distilbert-base-cased/cluster_300/concept_sentence.csv

python3.8 visualize_concept_with_sentence.py --concept_sentence_file distilbert-base-cased/cluster_300/concept_sentence.csv --concept_cluster_file distilbert-base-cased/cluster_300/clusters-300.txt --output_file distilbert-base-cased/cluster_300/visualization_layer6.pdf