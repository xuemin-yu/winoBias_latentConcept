import ast
from argparse import ArgumentParser

import pandas as pd

def read_csv(path):
    df = pd.read_csv(path, sep='\t')
    return df


def read_txt(path):
    with open(path) as f:
        lines = f.readlines()
    return lines


def filter_gender_related_MASK_prediction(masked_info_df):
    # filter out the 'embedding' columns
    masked_info_df = masked_info_df[masked_info_df.columns.drop(list(masked_info_df.filter(regex='embedding')))]

    # filter out the line that contains the word in gender_list
    gender_list = ["she", "She", "he", "He", "her", "Her", "his", "His", "him", "Him"]

    # get the 'tokens' column
    predicted_results = masked_info_df['predicted_results'].tolist()

    index = []
    for i in range(len(predicted_results)):
        if predicted_results[i] in gender_list:
            index.append(i)

    # only keep the rows that are in the index
    new_masked_info_df = masked_info_df.iloc[index]

    return new_masked_info_df


def match_MASK_prediction_with_concept(new_masked_info_df, masked_concept_df):
    # get the 'tokens' column
    tokens = new_masked_info_df['token'].tolist()

    index = []
    for i, token in enumerate(tokens):
        if token in masked_concept_df['Token'].tolist():
            idx = int(token[6:])
            index.append(idx)

    new_masked_concept_df = masked_concept_df.iloc[index]

    return new_masked_concept_df


def get_gender_related_concept(masked_concept_df):
    concepts = masked_concept_df['Top 1'].tolist()
    # get unique concepts
    gender_related_concepts = list(set(concepts))
    return gender_related_concepts


def get_sentences_by_concept(masked_concept_df, concept):
    new_df = masked_concept_df[masked_concept_df['Top 1'] == concept]
    sentences_idx = new_df['line_idx'].tolist()
    return sentences_idx

def get_MASK_prediction(sentence_idx, masked_info_df):
    sentence = masked_info_df.loc[masked_info_df['line_idx'] == sentence_idx]
    position_idx = int(sentence['position_idx'].values[0])
    predicted_result = sentence['predicted_results'].values[0]
    return position_idx, predicted_result


def get_all_sentences(data, sentences_idx, masked_info_df):
    sentences_list = []
    MASK_prediction_list = []
    for i in sentences_idx:
        position_idx, predicted_result = get_MASK_prediction(i, masked_info_df)
        sentence = data[i]
        sentence = sentence.replace('[MASK]', '['+ predicted_result + ']')
        sentences_list.append(sentence)
        MASK_prediction_list.append(predicted_result)
    return sentences_list, MASK_prediction_list


def stat_gender_in_a_concept(concept_sentence_df, concept):
    mask_prediction_list = concept_sentence_df[concept_sentence_df['concept'] == concept]['MASK_prediction'].tolist()[0]

    female_gender_words = ["she", "She", "her", "Her"]
    male_gender_words = ["he", "He", "his", "His", "him", "Him"]

    female_count = 0
    male_count = 0
    others_count = 0
    total_count = 0
    for i in range(len(mask_prediction_list)):
        if mask_prediction_list[i] in female_gender_words:
            female_count += 1
        elif mask_prediction_list[i] in male_gender_words:
            male_count += 1
        else:
            others_count += 1
        total_count += 1

    female_percentage = round(female_count / total_count * 100, 2)
    male_percentage = round(male_count / total_count * 100, 2)
    others_percentage = round(others_count / total_count * 100, 2)
    return female_percentage, male_percentage, others_percentage


def get_all_stats(concept_sentence_df, output_file):
    gender_related_concepts = concept_sentence_df['concept'].tolist()

    stat = []

    for concept in gender_related_concepts:
        female_percentage, male_percentage, others_percentage = stat_gender_in_a_concept(concept_sentence_df, concept)
        stat.append((female_percentage, male_percentage, others_percentage))

    # add the column to the dataframe
    concept_sentence_df['statistic'] = stat
    concept_sentence_df.to_csv(output_file, index=False, sep='\t')


def save_sentences_with_concept(masked_info_df, masked_concept_df, gender_related_concepts, data_sentences):
    total_sentences_list = []
    total_Mask_prediction_list = []
    for concept in gender_related_concepts:
        sentences_idx = get_sentences_by_concept(masked_concept_df, concept)
        sentences_list, MASK_prediction_list = get_all_sentences(data_sentences, sentences_idx, masked_info_df)
        total_sentences_list.append(sentences_list)
        total_Mask_prediction_list.append(MASK_prediction_list)

    # save concept with sentence_list to a csv file
    df = pd.DataFrame({'concept':  gender_related_concepts, 'sentences': total_sentences_list, 'MASK_prediction': total_Mask_prediction_list})
    return df


def main():
    parser = ArgumentParser()
    parser.add_argument("--masked_info_file", type=str, required=True)
    parser.add_argument("--masked_concept_file", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    masked_info_df = read_csv(args.masked_info_file)
    masked_concept_df = read_csv(args.masked_concept_file)
    data_sentences = read_txt(args.data_file)

    new_masked_info_df = filter_gender_related_MASK_prediction(masked_info_df)
    new_masked_concept_df = match_MASK_prediction_with_concept(new_masked_info_df, masked_concept_df)
    gender_related_concepts = get_gender_related_concept(new_masked_concept_df)

    df = save_sentences_with_concept(masked_info_df, masked_concept_df, gender_related_concepts, data_sentences)
    get_all_stats(df, args.output_file)



if __name__ == "__main__":
    main()







