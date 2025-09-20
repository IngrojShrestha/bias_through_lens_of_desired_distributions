import csv
import torch
from transformers import BertForMaskedLM, BertTokenizer, DistilBertForMaskedLM, DistilBertTokenizer
from datasets import Dataset, load_dataset
from tqdm import tqdm
import random
import numpy as np
import re
import pandas as pd
import argparse
from pathlib import Path
import os
from unittest.mock import patch
import torch.nn.functional as F
import langid
from langdetect import detect, DetectorFactory

# Make langdetect deterministic
DetectorFactory.seed = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training script with hyperparameter tuning.")
    
    parser.add_argument("--seed", type=int, choices= [42, 52, 62, 72, 82], required=True, help="Random seed")
    parser.add_argument("--model", type=str, default="bert_base", help="Name of the model (default: bert_base).")
    parser.add_argument("--model_state",help="bias evaluation stage (before_debiasing, after_debiasing)", required=True)    
    parser.add_argument("--run", type=str, required=True, help="Run identifier (e.g., run1).")
    parser.add_argument("--desired_dist", type=str, choices=["equal", "real_world"], required=True, help="Desired distribution type.")

    parser.add_argument("--base_output_path",
                        help="base path to save output_seed_42/results (e.g., ../../output_seed_42/mlm)",
                        required=True)
    
    parser.add_argument("--gamma", type = float, default = None, help = "provide weight for MLM loss")

    args = parser.parse_args()
    
    # gamma is required only for real world after debiasing where we include MLM loss also
    if args.desired_dist=='real_world' and args.model_state=='after_debiasing' and args.gamma is None:
        parser.error("--gamma (weights to MLM loss) is required for real world distribution after debiasing")
    
    return args

def print_args(args):
    """Prints all parsed arguments in a structured format."""
    print("\n--- Parsed Arguments ---")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("-"*100)
    print()

def ensemble_is_english(text, min_confidence=0.9):
    '''
    We use langdetect only to ensure the text is English.
    '''
    # langid detection
    # langid_lang, langid_conf = langid.classify(text)
    
    # langdetect detection
    try:
        langdetect_lang = detect(text)
    except:
        langdetect_lang = "error"

    is_english = False

    # # Ensemble decision logic
    # if langid_lang == 'en' and langid_conf >= min_confidence and langdetect_lang == 'en':
    #     is_english = True
    # elif langid_lang == 'en' and langid_conf >= 0.99:
    #     is_english = True
    # return is_english
    
    return langdetect_lang == 'en'
    
def clean_text(text):
    text = text.strip()

    # Remove artifacts
    text = text.replace("@-@", "-")              # Replace tokenization artifact
    text = text.replace("<unk>", "")             # Remove unknown tokens
    text = text.replace('""', '"')               # Collapse double quotes
    text = re.sub(r'\s+', ' ', text)             # Collapse multiple spaces
    
    # Remove one or more leading and trailing quotes ONLY if they are at the outer edges
    text = re.sub(r'^"+', '', text)   # Remove leading quotes (e.g., " or "")
    text = re.sub(r'"+$', '', text)   # Remove trailing quotes (e.g., " or "")
    
    return text.strip()

def load_wiki_text_dataset():
    
    dataset_type = 'train' # train, test, validation
    
    dataset = load_dataset("wikitext", "wikitext-103-v1", split=dataset_type)

    # Define a function to keep only lines that do not start with "=",
    # are not empty, and contain at least 15 words after splitting and removing punctuation
    def filter_lines(example):
        
        text = example['text'].strip()
        
        # Filter out empty or whitespace-only lines
        if text == "":
            return False
    
        if text.startswith("="):
            return False
        
        if text.startswith('"'):  # Exclude lines starting with a quote
            return False
        
        words = text.split()
        
        if all(re.fullmatch(r'\W+', word) for word in words):
            return False
        
        # Remove punctuation from each word
        words_cleaned = [re.sub(r'[^\w\s]', '', word) for word in words]
        
        # Filter out words that become empty after removing punctuation
        words_cleaned = [word for word in words_cleaned if word.strip() != ""]
        
        # Check if the sentence contains fewer than 15 words after cleaning
        if len(words_cleaned) < 15:
            return False

        return True

    # Apply the filter and update the dataset in place
    filtered_dataset = dataset.filter(filter_lines)

    en_sentences = []
    
    non_en_sentences = []

    print("Processing filtered sentences...")
    
    # Collect the filtered sentences
    for index, sentence in enumerate(filtered_dataset):
        
        if (index + 1) % 1000 == 0:
            print(f"Processed: {index + 1}/{len(filtered_dataset)}")
        
        cleaned = clean_text(sentence["text"])
        
        if cleaned != "":
            if ensemble_is_english(cleaned):
                en_sentences.append({"text": cleaned})
            else:
                non_en_sentences.append({"text": cleaned})
        
        # sentences.append({'text': sentence['text']})

    dataset = Dataset.from_dict({"text": [d["text"] for d in en_sentences]})
    
    # saved cleaned sentences
    df_en = pd.DataFrame(en_sentences)
    print("Total WikiText english examples:", df_en.shape)
    
    # preserve the text exactly but avoid wrapping it with quotes
    df_en.to_csv(f"../../data/WikiText-103-{dataset_type}-en.tsv", sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")
    
    df_non_en = pd.DataFrame(non_en_sentences)
    df_non_en.to_csv(f"../../data/WikiText-103-{dataset_type}-non-en.tsv", sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")
    
    print("Saved WikiText-103 cleaned dataset!!!")
    
    return dataset

def load_model(args):
    
    if args.model_state == "before_debiasing":
        if args.model == 'bert_base':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        elif args.model == 'bert_large':
            tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
            model = BertForMaskedLM.from_pretrained('bert-large-uncased')
        elif args.model == 'distilbert':
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
        
        print("Loading pre-trained model")
        
    elif args.model_state == 'after_debiasing':

        print("Loading debiased model...")     
                
        if args.desired_dist== 'equal':
            debiased_model_path = f"../../debiased_llms/output_seed_{args.seed}/mlm/{args.model}/{args.run}/{args.model}_{args.desired_dist}/tuning/best_model/"
            
        elif args.desired_dist== 'real_world':
            debiased_model_path = f"../../debiased_llms/output_seed_{args.seed}/mlm/{args.model}/{args.run}/{args.model}_{args.desired_dist}/gamma_{args.gamma}/tuning/best_model/"
        
        else:
            raise ValueError("Failed to load model!")
        
        print("Loading model from:", debiased_model_path)
        
        if args.model in ['bert_base', 'bert_large']:
            tokenizer = BertTokenizer.from_pretrained(debiased_model_path)
            model = BertForMaskedLM.from_pretrained(debiased_model_path)
        elif args.model == 'distilbert':
            tokenizer = DistilBertTokenizer.from_pretrained(debiased_model_path)
            model = DistilBertForMaskedLM.from_pretrained(debiased_model_path)
            
    model.to(device)
    model.eval()
    
    return model, tokenizer

# Function to mask exactly 15% of tokens in the input text
def mask_tokens(inputs, tokenizer, mask_prob=0.15):
    labels = inputs.clone()

    # Create a special tokens mask to avoid masking special tokens like [CLS], [SEP], [PAD]
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=inputs.device)
    
    # Calculate number of tokens to mask excluding special tokens
    non_special_token_indices = ~special_tokens_mask
    total_non_special_tokens = non_special_token_indices.sum(dim=1)
    num_masked_tokens = torch.max(torch.tensor([1], device=inputs.device), (total_non_special_tokens * mask_prob).floor()).long()

    batch_size = inputs.shape[0]

    # Mask exactly 15% of the non-special tokens in each sentence
    for i in range(batch_size):
        # Get indices of non-special tokens
        non_special_indices = torch.where(non_special_token_indices[i])[0]
        if len(non_special_indices) < num_masked_tokens[i]:
            # If there are fewer non-special tokens than required to mask, adjust accordingly
            tokens_to_mask = non_special_indices
        else:
            # Randomly select the tokens to mask
            tokens_to_mask = non_special_indices[torch.randperm(len(non_special_indices))[:num_masked_tokens[i]]]

        # Apply the mask
        inputs[i, tokens_to_mask] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    labels[~non_special_token_indices] = -100  # Only compute loss on non-special masked tokens

    return inputs, labels, total_non_special_tokens, num_masked_tokens

def trace_mlm_losses(logits, labels, inputs):
    '''
    Print loss for each of masked token
    Then compute average loss across masked tokens
    
    This method is used to test whether we compute mlm loss overall masked tokens across all sentences.
    '''
    vocab_size = logits.shape[-1]

    # Compute raw per-token loss
    loss_per_token = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        reduction='none'
    ).view(inputs.shape)  # Shape: [batch_size, seq_len]

    masked_positions = labels != -100

    total_loss = loss_per_token[masked_positions].sum().item()
    avg_loss = loss_per_token[masked_positions].mean().item()

    print()
    print("Per-token loss (all tokens):", loss_per_token[0].tolist())
    print("Masked positions:", masked_positions[0].tolist())
    print("Loss on masked tokens only:", loss_per_token[0][masked_positions[0]].tolist())
    print(f"Total loss over masked tokens: {total_loss:.4f}")
    print(f"Average loss over masked tokens: {avg_loss:.4f}")
    print()
        
# calculate MLM loss
def compute_mlm_loss(model, inputs, labels):
    with torch.no_grad():
        # Move inputs and labels to the device (GPU or CPU)
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(input_ids=inputs, labels=labels)
        loss = outputs.loss
        
        # trace per-token and total loss for inspection
        # trace_mlm_losses(outputs.logits, labels, inputs)
        
    return loss.item()

# decode input tokens into sentences (showing [MASK] tokens)
def decode_tokens(input_ids, tokenizer):
    # Ensure [MASK] is not skipped by setting skip_special_tokens=False
    return tokenizer.decode(input_ids, skip_special_tokens=False)

# Tokenize input data, compute loss, and save original/masked texts
def calculate_mlm_loss_and_save(args, dataset, dataset_name, tokenizer, model, output_file_individual_mlm_loss, output_file_mlm_loss):
    
    total_loss = 0.0
    total_masked_tokens = 0
    total_original_tokens = 0  # This will hold the sum of all original tokens excluding [CLS] and [SEP]
    
    results = []
    
    for example in tqdm(dataset):
        
        text = example['text']
        
        # Tokenize the text batch
        tokenized_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        input_ids = tokenized_inputs.input_ids.to(device)  # Shape: [1, seq_len]
        
        input_ids_original = input_ids.clone() # we make a copy to get original tokens as we are making inplace replacement in mask_token() method
        
        tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs.input_ids[0])

        # Mask exactly 15% of tokens
        masked_input_ids, labels, total_non_special_tokens, num_masked_tokens = mask_tokens(input_ids, tokenizer)
        
        # keep track of total tokens across sentences
        total_original_tokens += total_non_special_tokens[0].item()
            
        # Compute loss for this single example
        loss = compute_mlm_loss(model, masked_input_ids, labels)

        masked_sentence = decode_tokens(masked_input_ids[0], tokenizer)

        results.append({
            "original_text": text,
            "original tokens": tokens,
            "original input_ids": input_ids_original[0].tolist(),
            "masked_sentence": masked_sentence,
            "masked token ids": masked_input_ids[0].tolist(), 
            "loss": loss,
            "num_tokens_excluding_special": total_non_special_tokens[0].item(),
            "num_masked_tokens": num_masked_tokens[0].item()
        })

        total_loss += loss * num_masked_tokens[0].item()  # Accumulate total weighted loss
        total_masked_tokens += num_masked_tokens[0].item()

    df = pd.DataFrame(results)
    df.to_csv(output_file_individual_mlm_loss, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")

    print(f"Individual MLM loss results saved to: {output_file_mlm_loss}")
    
    mlm_loss = total_loss / total_masked_tokens
    
    with open(output_file_mlm_loss, "w") as f:
        f.write("-" * 100 + "\n")
        
        f.write("Dataset: " + dataset_name + "\n")
        f.write("-" * 100 + "\n")
        
        f.write("Model: " + args.model + "\n")
        f.write("Model state: " + args.model_state + "\n")
        f.write("Run: " + args.run + "\n")
        f.write("Desired distribution: " + args.desired_dist + "\n")
        f.write("Gamma: " + str(args.gamma) + "\n")
        f.write("-" * 100 + "\n")
        
        f.write("Total sentences processed: " + str(len(dataset)) + "\n")
        f.write("Total original tokens (excluding special tokens): " + str(total_original_tokens) + "\n")
        f.write("Total masked tokens: " + str(total_masked_tokens) + "\n")
        f.write("Aggregate MLM loss: " + str(mlm_loss) + "\n")

def set_paths(args, dataset_name):
    
    print("setting paths....")
    
    if args.desired_dist == 'real_world':
        
        if args.model_state == 'before_debiasing':
            BASE_PATH = Path(args.base_output_path) / args.model / args.run / f"{args.model}_real_world" / args.model_state

        elif args.model_state == 'after_debiasing':
            BASE_PATH = Path(args.base_output_path) / args.model / args.run / f"{args.model}_real_world" / f"gamma_{args.gamma}"/ args.model_state
            
        else:
            raise ValueError ('Invalid model_state')
            
    elif args.desired_dist == 'equal':
        BASE_PATH = Path(args.base_output_path)  / args.model / args.run / f"{args.model}_equal" / args.model_state

    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
            
    INDIVIDUAL_MLM_LOSS_OUTPUT_PATH = BASE_PATH / f"mlm_loss_{dataset_name}_individual_results_{args.run}_{args.desired_dist}_{args.model_state}.tsv"
    MLM_LOSS_OUTPUT_PATH = BASE_PATH / f"mlm_loss_{dataset_name}_total_results_{args.run}_{args.desired_dist}_{args.model_state}.txt"
    
    return INDIVIDUAL_MLM_LOSS_OUTPUT_PATH, MLM_LOSS_OUTPUT_PATH

def load_data():
    
    df_wikitext_103_test = pd.read_csv("../../data/WikiText-103-test-en.tsv", sep="\t")
    df_wikitext_103_valid = pd.read_csv("../../data/WikiText-103-validation-en.tsv", sep="\t")
    df_gap_corpus = pd.read_csv("../../data/gap_flipped.tsv", sep="\t")
    
    text_df_wikitext_103_test = df_wikitext_103_test[['text']].assign(text=lambda x: x['text'].str.lower())
    text_df_wikitext_103_valid  = df_wikitext_103_valid[['text']].assign(text=lambda x: x['text'].str.lower())
    text_df_gap_corpus = df_gap_corpus[['text']].assign(text=lambda x: x['text'].str.lower())
    
    dataset_wikitext_103_test = Dataset.from_pandas(text_df_wikitext_103_test, preserve_index=False)
    dataset_wikitext_103_valid = Dataset.from_pandas(text_df_wikitext_103_valid, preserve_index=False)
    dataset_gap_corpus = Dataset.from_pandas(text_df_gap_corpus, preserve_index=False)
    
    return dataset_wikitext_103_test, dataset_wikitext_103_valid, dataset_gap_corpus
    
    
def main():
    
    args = parse_arguments()
    
    print_args(args)
    
    set_seed(args.seed) 
    
    # to generate wikiText-103 clean dataset
    # dataset = load_wiki_text_dataset()
    
    model, tokenizer = load_model(args)
    
    dataset_wikitext_103_test, dataset_wikitext_103_valid, dataset_gap_corpus = load_data()
    
    datasets = [dataset_wikitext_103_test, dataset_wikitext_103_valid, dataset_gap_corpus]
    labels = ["WikiText-103-test", "WikiText-103-validation", "Gap Corpus"]
    
    for i in range(len(datasets)):
        dataset = datasets[i]
        dataset_name = labels[i]
        
        print()
        INDIVIDUAL_MLM_LOSS_OUTPUT_PATH, MLM_LOSS_OUTPUT_PATH = set_paths(args, dataset_name)
        print("Processing dataset: ", dataset_name)
        calculate_mlm_loss_and_save(args, dataset, dataset_name, tokenizer,model, INDIVIDUAL_MLM_LOSS_OUTPUT_PATH, MLM_LOSS_OUTPUT_PATH)
    
if __name__=="__main__":
    main()

