""" Generate the explanation file for [CLS] tokens.

This scripts will get the predicted label for each sentence in the dataset, and then use the predicted label,
sentence_idx and word_idx to generate the explanation file for CLS tokens.
"""

import argparse

import pandas as pd
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn.functional as F
import os
import json


def get_dataset(file_name):
    with open(file_name, 'r') as f:
        sentences = f.readlines()
    return sentences


# Get the mask token index and value
def get_mask_id_val(inputs, logits, tokenizer, index):
    # retrieve index of [MASK]
    mask_token_index = (inputs.input_ids[index] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    predicted_token_id = logits[index][mask_token_index].argmax(axis=-1)
    predicted_result = tokenizer.decode(predicted_token_id)

    return predicted_result, mask_token_index

def get_hidden_states_inputs(model, tokenizer, sentence, device):
    """
    Input the sentences into the model to get the hidden states and predicted label of the model.

    Parameters
    ----------
    model: transformers.modeling_utils.PreTrainedModel
        The pre-trained model to be used.

    tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase
        The tokenizer to be used.

    dataset: datasets.arrow_dataset.Dataset
        The dataset to be tokenized.
    """

    inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    logits = outputs.logits

    results = []
    mask_idx = []

    for i in range(len(inputs['input_ids'])):
        predicted_result, mask_token_index = get_mask_id_val(inputs, logits, tokenizer, i)
        results.append(predicted_result)
        mask_idx.append(mask_token_index)

    return inputs, results, mask_idx, outputs.hidden_states


def save_mask_token(attention_hidden_states, mask_idx, results):
    if not os.path.exists('masked_representation_info'):
        os.makedirs('masked_representation_info')

    for i in range(7):# 6 layers

        tokens = []
        line_idx = []
        position_idx = []
        embedding = []
        predicted_results = []

        for j in range(len(mask_idx)): # len(ids) => # of sentences
          # get the data representation of a layer for a sentence
          layer_content = attention_hidden_states[i][j]

          tokens.append("[MASK]" + str(j))
          line_idx.append(j)
          position_idx.append(mask_idx[j].item())
          embedding.append(layer_content[mask_idx[j]].tolist()[0])
          predicted_results.append(results[j])

          # get the [MASK] representation and info
          # info = []
          # info.append("[MASK]" + "|||" + str(mask_idx[j].item()) + "|||" + str(j))
          # info.append(layer_content[mask_idx[j]].tolist()[0])
          # info.append(results[j])
          #
          # layer_info.append(info)

        # save all [MASK] embedding by each layer in a csv file

        # with open('masked_representation_info/masked_representation_'+str(i)+'.json', 'w') as json_file:
        #   json.dump(layer_info, json_file, indent=4)
        df = pd.DataFrame({'token': tokens, 'line_idx': line_idx, 'position_idx': position_idx, 'embedding': embedding, 'predicted_results': predicted_results})
        
        if not os.path.exists('../masked_representation_info'):
            os.makedirs('../masked_representation_info')
        df.to_csv('../masked_representation_info/masked_representation_'+str(i)+'.csv', sep='\t', index=False)

# def save_prediction(predicted_result, sentence, output_path):
#     path = output_path + '/predicted_result.csv'
#
#     df = pd.DataFrame({'sentence': sentence, 'predicted_result': predicted_result})
#     df.to_csv(path, sep='\t', index=False)


def generate_explanation(ids, predicted_result, mask_idx, save_path):
    """
    Generate and save the explanation for the [CLS] tokens.

    Parameters
    ----------
    ids: torch.Tensor
        The token ids of the sentences.

    labels: List
        The predicted labels of the sentences.

    save_path: str
        The path to save the json file.

    Returns
    -------
    layer_cls_info: list
        The [CLS] token representation and info.
    """
    predictions = []

    for j in range(len(ids)):  # len(ids) => # of sentences
        # prediction class(label) ||| position_id ||| sentence_id

        predictions.append(str(predicted_result[j]) + " " + str(mask_idx[j].item()) + " " + str(j))

    # check the directory exists or not
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    path = save_path + '/explanation_mask_token.txt'
    # save the explanation in a txt file
    with open(path, "w") as txt_file:
        for line in predictions:
            txt_file.write(line + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name-or-path',
                        type=str,
                        default='./data/test.txt',
                        help='The name or path of the dataset to be loaded.')

    parser.add_argument('--model-name',
                        type=str,
                        default='bert-base-cased',
                        help='The name or path of the pre-trained model to be used.')

    parser.add_argument('--tokenizer-name',
                        type=str,
                        default='bert-base-cased',
                        help='The name or path of the tokenizer to be used.')

    parser.add_argument('--save-dir',
                        type=str,
                        default='masked_tokens/',
                        help='The directory to save the extracted masked token representation and info.')

    args = parser.parse_args()

    sentence = get_dataset(args.dataset_name_or_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name).to(device)

    print("Finishing loading the dataset, model and tokenizer.")

    inputs, results, mask_idx, attention_hidden_states = get_hidden_states_inputs(model, tokenizer, sentence, device)
    print("Finishing getting the hidden states and predicted labels.")

    save_mask_token(attention_hidden_states, mask_idx, results)
    # save_prediction(results, sentence,  args.save_dir)

    ids = inputs['input_ids']

    generate_explanation(ids, results, mask_idx, args.save_dir)
    print("Finishing generating the explanation for [MASK] tokens.")


if __name__ == '__main__':
    main()