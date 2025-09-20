import torch.nn.functional as F
import pandas as pd
import os
import warnings
import functools
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import Dataset
import csv

access_token = "HUGGING_FACE_TOKEN"

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

print = functools.partial(print, flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def parse_arguments():
        
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", 
                        type=int, 
                        choices= [42, 52, 62, 72, 82],
                        required=True, 
                        help="Random seed")
    
    parser.add_argument("--model",
                        type = str, 
                        choices = ['llama3.2_3B_Instruct',
                                   'llama3.1_8B_Instruct','llama3.3_70B_Instruct',
                                   'qwen2.5_7B_Instruct','qwen2.5_72B_Instruct'],
                        help = "model name (e.g., llama3.2_3B)",
                        required=True)
    
    parser.add_argument("--run",
                        help="run number e.g., run1")
    
    parser.add_argument("--desired_dist",
                        help="desired distribution (equal, real_world)")
    
    parser.add_argument("--model_state",
                        help="bias evaluation stage (before_debiasing, after_debiasing)",
                        required=True)
    
    parser.add_argument("--base_output_path",
                        help="base path to save output_seed_42/results (e.g., ../../output_seed_42/alm)",
                        required=True)
    
    args = parser.parse_args()
    
    if args.model == 'llama3.2_3B_Instruct':
        args.model_id = "meta-llama/Llama-3.2-3B-Instruct"
        
    elif args.model == 'llama3.1_8B_Instruct':
        args.model_id = "meta-llama/Llama-3.1-8B-Instruct"
        
    elif args.model == 'llama3.3_70B_Instruct':
        args.model_id = "meta-llama/Llama-3.3-70B-Instruct"
    
    elif args.model =='qwen2.5_7B_Instruct':
        args.model_id = "Qwen/Qwen2.5-7B-Instruct"
        
    elif args.model =='qwen2.5_72B_Instruct':
        args.model_id = "Qwen/Qwen2.5-72B-Instruct"
        
    return args

############################################################################################################
args = parse_arguments()

def print_args(args):
    
    print("\n--- Parsed Arguments ---")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("-"*100)
    print()
    print("device:", device) 
    print("-" * 100)
    print()
    
print_args(args)
############################################################################################################

def load_model(args):
    if args.model_state == 'before_debiasing':
        print("Loading pretrained model...")
        
        quantized_models = {
            'llama3.1_8B_Instruct',
            'llama3.3_70B_Instruct',
            'qwen2.5_7B_Instruct',
            'qwen2.5_72B_Instruct'
        }
        
        qwen_models = {
            'qwen2.5_7B_Instruct',
            'qwen2.5_72B_Instruct'
        }
        
        if args.model in ['llama3.2_3B_Instruct']:
            tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_auth_token=access_token)
            model = AutoModelForCausalLM.from_pretrained(args.model_id, use_auth_token=access_token)

        # for rest of the model use quantization for inference also
        elif args.model in quantized_models: 
            quant_config = BitsAndBytesConfig(load_in_4bit=True,
                                                bnb_4bit_quant_type="nf4",
                                                bnb_4bit_use_double_quant=True,
                                                bnb_4bit_compute_dtype=torch.float16)
            
            # for qwen models
            if args.model in qwen_models:
                tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(args.model_id,
                                                                trust_remote_code=True,
                                                                device_map="auto", # dont use as it creates problem
                                                                quantization_config=quant_config)
            # for llama3.1 larger models
            else:
                tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_auth_token=access_token)
                model = AutoModelForCausalLM.from_pretrained(args.model_id, 
                                                                use_auth_token=access_token,
                                                                device_map="auto", # dont use as it creates problem
                                                                quantization_config=quant_config)
        else:
            raise ValueError(f"Unrecognized model name: {args.model}")   

    elif args.model_state == 'after_debiasing':
        print("Loading debiased model...")     
        
        debiased_adapter_path = f"../../debiased_llms/output_seed_{args.seed}/alm/{args.model}/{args.run}/{args.model}_{args.desired_dist}/tuning/best_model/"
        
        print("Loading model from:", debiased_adapter_path)
        
        quantized_models = {
            'llama3.1_8B_Instruct',
            'llama3.3_70B_Instruct',
            'qwen2.5_7B_Instruct',
            'qwen2.5_72B_Instruct'
        }

        qwen_models = {
            'qwen2.5_7B_Instruct',
            'qwen2.5_72B_Instruct'
        }

        # LoRA (non-quantized) model
        if args.model in ['llama3.2_3B_Instruct']:
            
            tokenizer = AutoTokenizer.from_pretrained(debiased_adapter_path)

            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                # device_map="auto",               # auto-loads across available GPUs
                trust_remote_code=True,
                torch_dtype=torch.float16,  # or torch.bfloat16 if supported
            )

            model = PeftModel.from_pretrained(base_model, debiased_adapter_path) # device_map="auto" (avoid using auto)

        # QLoRA models
        elif args.model in quantized_models:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )

            if args.model in qwen_models:
                tokenizer = AutoTokenizer.from_pretrained(debiased_adapter_path, trust_remote_code=True)
                base_model = AutoModelForCausalLM.from_pretrained(
                    args.model_id,
                    trust_remote_code=True,
                    quantization_config=quant_config,
                    device_map="auto"
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(debiased_adapter_path, use_auth_token=access_token)
                base_model = AutoModelForCausalLM.from_pretrained(
                    args.model_id,
                    use_auth_token=access_token,
                    quantization_config=quant_config,
                    device_map="auto"
                )

            model = PeftModel.from_pretrained(base_model, debiased_adapter_path)
        else:
            raise ValueError(f"Unknown model name: {args.model}")

    # if args.model not in ['llama3.2_3B_Instruct']:
    #     pass
    # else:
    #     model.to(device)
    
    # set both pretrained or loaded debiased model in evaluation mode
    model.eval()
    
    return model, tokenizer

def calculate_corpus_perplexity_and_save(args, dataset, dataset_name, tokenizer, model, output_file_individual_ppl, output_file_overall_ppl):
    # if args.model not in ['llama3.2_3B_Instruct']:
    #     pass
    # else:
    #     model.to(device)
        
    model.eval()

    results = []
    
    total_loss = 0.0
    total_tokens = 0

    for example in tqdm(dataset):
        
        text = example['text']
        
        # if args.model not in ['llama3.2_3B_Instruct']:
        #     tokens = tokenizer(text, return_tensors='pt')
        # else:
        #     tokens = tokenizer(text, return_tensors='pt').to(device)
        
        tokens = tokenizer(text, return_tensors='pt')
        
        input_ids = tokens['input_ids']
        
        token_ids = input_ids.squeeze().tolist()
        token_strs = tokenizer.convert_ids_to_tokens(token_ids)

        with torch.no_grad():
            outputs = model(**tokens, labels=tokens['input_ids'])
            # outputs = model(input_ids=input_ids, labels=input_ids)
        
            loss = outputs.loss
            num_tokens = input_ids.numel()
            
            cur_ppl = torch.exp(loss)

        results.append({
            "original_text": text, 
            "token_ids": token_ids,
            "token_strings": token_strs,
            "perplexity": cur_ppl.item(),
            "num_tokens": num_tokens
        })
        
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    df = pd.DataFrame(results)
    df.to_csv(output_file_individual_ppl, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")
    
    print(f"Individual Perplexity results saved to: {output_file_individual_ppl}")
    
    avg_perplexity = math.exp(total_loss / total_tokens)
    
    with open(output_file_overall_ppl, "w") as f:
        f.write("-" * 100 + "\n")
        
        f.write("Dataset: " + dataset_name + "\n")
        f.write("-" * 100 + "\n")
        
        f.write("Model: " + args.model + "\n")
        f.write("Model state: " + args.model_state + "\n")
        f.write("Run: " + args.run + "\n")
        f.write("Desired distribution: " + args.desired_dist + "\n")
        f.write("-" * 100 + "\n")
        
        f.write("Total sentences processed: " + str(len(dataset)) + "\n")
        f.write("Total original tokens: " + str(total_tokens) + "\n")
        f.write("Aggregate Perplexity: " + str(avg_perplexity) + "\n")

def load_data():
    df_wikitext_103_test = pd.read_csv("../../data/WikiText-103-test-en.tsv", sep="\t")
    df_wikitext_103_valid = pd.read_csv("../../data/WikiText-103-validation-en.tsv", sep="\t")
    df_gap_corpus = pd.read_csv("../../data/gap_flipped.tsv", sep="\t")
    
    text_df_wikitext_103_test = df_wikitext_103_test[['text']]
    text_df_wikitext_103_valid  = df_wikitext_103_valid[['text']]
    text_df_gap_corpus = df_gap_corpus[['text']]
    
    dataset_wikitext_103_test = Dataset.from_pandas(text_df_wikitext_103_test, preserve_index=False)
    dataset_wikitext_103_valid = Dataset.from_pandas(text_df_wikitext_103_valid, preserve_index=False)
    dataset_gap_corpus = Dataset.from_pandas(text_df_gap_corpus, preserve_index=False)
    
    return dataset_wikitext_103_test, dataset_wikitext_103_valid, dataset_gap_corpus

def set_paths(args, dataset_name):
    
    print("setting paths....")
    
    if args.desired_dist == 'real_world':
        BASE_PATH = Path(args.base_output_path) / args.model / args.run / f"{args.model}_real_world" / args.model_state
        
    elif args.desired_dist == 'equal':
        BASE_PATH = Path(args.base_output_path)  / args.model / args.run / f"{args.model}_equal" / args.model_state

    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
            
    INDIVIDUAL_PPL_OUTPUT_PATH = BASE_PATH / f"ppl_{dataset_name}_individual_results_{args.run}_{args.desired_dist}_{args.model_state}.tsv"
    OVERALL_PPL_OUTPUT_PATH = BASE_PATH / f"ppl_{dataset_name}_total_results_{args.run}_{args.desired_dist}_{args.model_state}.txt"
    
    return INDIVIDUAL_PPL_OUTPUT_PATH, OVERALL_PPL_OUTPUT_PATH

def main():
        
    model, tokenizer = load_model(args)
    
    dataset_wikitext_103_test, dataset_wikitext_103_valid, dataset_gap_corpus = load_data()
    
    datasets = [dataset_wikitext_103_test, dataset_wikitext_103_valid, dataset_gap_corpus]
    labels = ["WikiText-103-test", "WikiText-103-validation", "Gap Corpus"]
    
    for i in range(len(datasets)):
        dataset = datasets[i]
        dataset_name = labels[i]
        
        print()
        INDIVIDUAL_PPL_OUTPUT_PATH, OVERALL_PPL_OUTPUT_PATH = set_paths(args, dataset_name)
        print("Processing dataset: ", dataset_name)
        calculate_corpus_perplexity_and_save(args, dataset, dataset_name, tokenizer,model, INDIVIDUAL_PPL_OUTPUT_PATH, OVERALL_PPL_OUTPUT_PATH)
    
if __name__ == "__main__":
    main()
