'''
Bias evaluation in Llama-3.2-3B-Instruct, Llma-3.1-8B-Instruct, Llma-3.3-70B-Instruct, Qwen2.5-7B-Instruct, Qwen2.5-72B-Instruct for both equal and real-world desired distribution.
'''
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import warnings
import functools
import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

access_token = "HUGGING_FACE_TOKEN"

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

print = functools.partial(print, flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (True/False).")
    
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
    
    parser.add_argument("--batchmean",
                    type = str2bool,
                    help="Batch mean (True, False)",
                    required=True)
    
    parser.add_argument("--run",
                        help="run number e.g., run1")
    
    parser.add_argument("--eval_set",
                        help="evaluation set (train, valid, test)")
    
    parser.add_argument("--desired_dist",
                        help="desired distribution (equal, real_world)")
    
    parser.add_argument("--model_state",
                        help="bias evaluation stage (before_debiasing, after_debiasing)",
                        required=True)
    
    parser.add_argument("--base_output_path",
                        help="base path to save output_seed_42/results and (e.g., ../../output_seed_42/alm)",
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
        
    # base directory
    args.data_dir = Path("../../data")
    
    # Add formatted paths to args
    args.target_file = args.data_dir / f"professions_{args.eval_set}.tsv"
    

    if args.eval_set in ['train', 'valid']: # use same templates for training and validation
        args.template_path = args.data_dir / f"templates_train.tsv"
    elif args.eval_set == 'test': 
        args.template_path = args.data_dir / f"templates_test.tsv"

    args.male_gendered_words_path = args.data_dir / "male_gendered_words.txt"
    args.female_gendered_words_path = args.data_dir / "female_gendered_words.txt"
    
    return args

############################################################################################################
args = parse_arguments()

if args.desired_dist == 'real_world':

    df_dist = pd.read_csv(args.data_dir / "profession_dist.tsv", sep="\t")
    real_dist = df_dist.set_index('profession')[['female_dist', 'male_dist']].T.to_dict()
    
    BASE_PATH = Path(args.base_output_path) / args.model / args.run / f"{args.model}_real_world" / args.model_state
        
elif args.desired_dist == 'equal':
    BASE_PATH = Path(args.base_output_path)  / args.model / args.run / f"{args.model}_equal" / args.model_state

if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)
        
NORMALIZED_BY_PROFESSION_PATH = BASE_PATH / f"normalized_score_by_profession_{args.model_state}_{args.eval_set}.tsv"
RAW_ASSOCIATION_SCORE_PATH = BASE_PATH / f"raw_association_scores_{args.model_state}_{args.eval_set}.tsv"

def print_args(args):
    
    print("-" * 100)
    print("device:", device)
    print("seed:", args.seed)
    print("model: ", args.model)
    print("model_id: ", args.model_id)
    print("batchmean: ", args.batchmean)
    print("run: ", args.run)
    print("eval_set: ", args.eval_set)
    print("desired_dist: ", args.desired_dist)
    print("model_state: ", args.model_state)
    print("base_output_path: ", args.base_output_path)
    print("data_dir: ", args.data_dir)
    print("target_file: ", args.target_file)
    print("template_path: ", args.template_path)
    print("male_gendered_words_path: ", args.male_gendered_words_path)
    print("female_gendered_words_path: ", args.female_gendered_words_path)
    print("BASE_PATH:", BASE_PATH)
    print("NORMALIZED_BY_PROFESSION_PATH:", NORMALIZED_BY_PROFESSION_PATH)
    print("RAW_ASSOCIATION_SCORE_PATH:", RAW_ASSOCIATION_SCORE_PATH)
    
    print("-" * 100)
    print()
    
print_args(args)
############################################################################################################

def is_vowel(word): 
    import eng_to_ipa as ipa

    # CMU to IPA notations
    symbols = {"a": "ə", "ey": "eɪ", "aa": "ɑ", "ae": "æ", "ah": "ə", "ao": "ɔ",
               "aw": "aʊ", "ay": "aɪ", "ch": "ʧ", "dh": "ð", "eh": "ɛ", "er": "ər",
               "hh": "h", "ih": "ɪ", "jh": "ʤ", "ng": "ŋ", "ow": "oʊ", "oy": "ɔɪ",
               "sh": "ʃ", "th": "θ", "uh": "ʊ", "uw": "u", "zh": "ʒ", "iy": "i", "y": "j"}

    vowels = ["ə", "eɪ", "ɑ", "æ", "ə", "ɔ", "aʊ", "aɪ", "ɛ", "ər", "ɪ", "oʊ", "ɔɪ", "u", "i"]

    phoneme = ipa.convert(word)
    phoneme = phoneme.replace("ˈ", '').replace("ˌ", '')
    if phoneme[0] in vowels:
        return True
    else:
        return False

def get_attributes(male_attribute_path, female_attribute_path):

    def read_file(filepath):
        with open(filepath) as f:
            return [line.strip() for line in f.readlines()]
        
    gender_dict = {name: "male" for name in read_file(male_attribute_path)}
    gender_dict.update({name: "female" for name in read_file(female_attribute_path)})
   
    return gender_dict

def get_templates(template_path):
    
    df = pd.read_csv(template_path, sep="\t")
    templates = dict(zip(df['template_id'], df['template']))
    return templates

def get_targets(target_file):
    df_professions = pd.read_csv(target_file, sep="\t")
    return df_professions['profession'].values.tolist()

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
        
        # do not use quantization for llama3.2-3B-Instruct
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
                                                                # device_map="auto", # dont use as it creates problem
                                                                quantization_config=quant_config)
            # for llama3.1 larger models
            else:
                tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_auth_token=access_token)
                model = AutoModelForCausalLM.from_pretrained(args.model_id, 
                                                                use_auth_token=access_token,
                                                                # device_map="auto", # dont use as it creates problem
                                                                quantization_config=quant_config)
        else:
            raise ValueError(f"Unrecognized model name: {args.model}")   
            
    elif args.model_state == 'after_debiasing':
        
        print("Loading debiased model...")     
        
        # only for real_world (as there is no bias for equal distribution)
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
    
    # model.to(device)
    
    # set both pretrained or loaded debiased model in evaluation mode
    model.eval()
    
    return model, tokenizer

def extract_scores(df, professionOrder, gender):
    # Filter the DataFrame for male gender
    df_filtered = df[df['gender'] == gender]

    # Directly set profession as a categorical with the specified order from profOrder
    df_filtered['profession'] = pd.Categorical(df_filtered['profession'], categories=professionOrder, ordered=True)

    # Sort the DataFrame based on the order in profOrder
    sorted_filtered_df = df_filtered.sort_values('profession')

    return sorted_filtered_df['softmax_score'].values.tolist()

def get_normalized_by_profession(df):
    # Sum exponentials by profession and gender
    exp_sums = df.groupby(['profession', 'gender'])['exp_score'].sum().reset_index()

    # Compute total exponentials sum per profession
    exp_sums['total_exp'] = exp_sums.groupby('profession')['exp_score'].transform('sum')

    # Normalize by dividing gender-specific exponential sum by the total per profession
    exp_sums['softmax_score'] = exp_sums['exp_score'] / exp_sums['total_exp']

    df_normalized_by_profession = exp_sums[['profession', 'gender', 'softmax_score']]
    
    df_normalized_by_profession.to_csv(NORMALIZED_BY_PROFESSION_PATH, sep = "\t", index=False)
    
    # formate to the structure of {'doctor' :{'male': softmax_score, 'female': softmax_score}}
    professions_dict = {}
    for profession, group in df_normalized_by_profession.groupby('profession'):
        professions_dict[profession] = {
            'male': group[group['gender'] == 'male']['softmax_score'].values[0],
            'female': group[group['gender'] == 'female']['softmax_score'].values[0]
        }

    return professions_dict

def calculate_loss_and_perplexity(model, tokenizer, sentence):
    
    tokens = tokenizer(sentence, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**tokens, labels=tokens['input_ids'])
    
    loss = outputs.loss.item()
    perplexity = torch.exp(outputs.loss).item()

    return loss, perplexity

def get_gender_profession_distribution(model, tokenizer, templates,gender_dict, professions):
    
    df = pd.DataFrame()
    
    association_scores  = {profession: {'male': [], 'female': []} for profession in professions}
    
    for template_id, template in templates.items():
        
        template_actual = template
        
        for target in professions:
            
            print(f"Template: '{template_id}', profession: {target}")
            
            for gender_value, gender in gender_dict.items():
                
                template = template_actual
                
                # extract only the gendered value (exclude determiner/pronoun)
                gendered_word_with_det = gender_value.split()
                
                if len(gendered_word_with_det) > 1:
                    gender_value_ = " ".join(gendered_word_with_det[1:])
                    determiner = gendered_word_with_det[0] + " "
                else:
                    gender_value_ = gendered_word_with_det[0]
                    determiner = ""
                
                if is_vowel(target):
                    template = template.replace("[ARTICLE]", "an")
                else:
                    template = template.replace("[ARTICLE]", "a")                
                
                # add determiner to the template from the gender_value
                template = template.replace("[DET]", determiner)
                
                original_sentence  = template.replace("[AAA]", gender_value_).replace("[TTT]", target)

                sentence_loss, perplexity = calculate_loss_and_perplexity(model, tokenizer, original_sentence)
               
                temp_dict = {'template_id': template_id, 
                             'template': template_actual, 
                             'gendered_word': gender_value_,
                             'gender': gender,
                             'profession': target,
                             'sentence': original_sentence,
                             'association_score': sentence_loss, # already used .item()
                             'perplexity': perplexity # already used .item()
                             }
                
                association_scores[target][gender].append(sentence_loss)
                
                df_temp = pd.DataFrame([temp_dict])
                df = pd.concat([df, df_temp], ignore_index=True)
    
    # Compute exponentials of the scores
    df['exp_score'] = np.exp(-df['association_score']) # higher loss = weaker association percentage. so use e^(-x) instead of e^x
    
    df.to_csv(RAW_ASSOCIATION_SCORE_PATH, sep = "\t", index=False) 
    
    # male_scores, female_scores save in dataframe
    return get_normalized_by_profession(df)

def compute_kl(pred, true):
    
    epsilon = 1e-8
    
    true_m, true_f = true
    pred_m, pred_f = pred
    
    # True distribution (target) and predicted distribution
    true_distribution = torch.tensor([true_m, true_f], dtype=torch.float32)  # Example true [m, f]
    predicted_distribution = torch.tensor([pred_m, pred_f], dtype=torch.float32)  # Example pred [m, f]

    # Clamp predicted distribution between epsilon and 1.0
    predicted_distribution = torch.clamp(predicted_distribution, min=epsilon, max=1.0)

    # Compute log of predicted distribution directly (since it's already in [0, 1])
    predicted_log_prob = torch.log(predicted_distribution)

    # Compute individual KL divergences for male and female
    kl_div_m = F.kl_div(predicted_log_prob[0].unsqueeze(0), true_distribution[0].unsqueeze(0), reduction='sum')
    kl_div_f = F.kl_div(predicted_log_prob[1].unsqueeze(0), true_distribution[1].unsqueeze(0), reduction='sum')
    
    # # Compute total KL divergence
    if args.batchmean:
        kl_div_total = F.kl_div(predicted_log_prob, true_distribution, reduction='batchmean')
    else:
        kl_div_total = F.kl_div(predicted_log_prob, true_distribution, reduction='sum')

    return kl_div_total.item(), kl_div_m.item(), kl_div_f.item()

def compute_kl_by_prof_gender_ALL(df):
    kl_lst = []
    
    # this is KL considering both male and female (we take average in our case)
    df['KL_female'] = ''
    df['KL_male'] = ''
    df['KL_total'] = ''
    
    
    for profession in df['profession'].values.tolist():
        
        if args.model_state == 'after_debiasing':
            
            pred_dist_male = df[df['profession'] == profession]['m after'].values[0]
            pred_dist_female = df[df['profession'] == profession]['f after'].values[0]
        
        elif args.model_state == 'before_debiasing':
            
            pred_dist_male = df[df['profession'] == profession]['m before'].values[0]
            pred_dist_female = df[df['profession'] == profession]['f before'].values[0]
            
        pred = [pred_dist_male, pred_dist_female]
        
        if args.desired_dist =='equal':
            true = [0.5, 0.5]
        elif args.desired_dist == 'real_world':
            true_dist_male = df[df['profession'] == profession]['m true'].values[0]
            true_dist_female= df[df['profession'] == profession]['f true'].values[0]
            true = [true_dist_male, true_dist_female]
        
        cur_kl_div_total, cur_kl_div_m, cur_kl_div_f = compute_kl(pred, true)
        kl_lst.append(cur_kl_div_total)

        df.loc[df['profession']==profession, 'KL_female'] = cur_kl_div_f
        df.loc[df['profession']==profession, 'KL_male'] = cur_kl_div_m
        df.loc[df['profession']==profession, 'KL_total'] = cur_kl_div_total # average of male and female kl deviation from true distribution
        
    
    category_stats = df.groupby('prof_gender')['KL_total'].agg(KL_mean='mean', KL_variance='var').reset_index()
    overall_stats = pd.DataFrame({'prof_gender': ['ALL'], 'KL_mean': [df['KL_total'].mean()], 'KL_variance': [df['KL_total'].var()]})
    df_summary = pd.concat([overall_stats, category_stats], ignore_index=True)
    
    return df, df_summary

def get_kl_individual_and_summary(predicted_distribution):
    # extract individual male and female distribution (true and predicted) for each profession
    # also compute corresponding KL divergence
    # finally compute KL mean and KL variance for each prof_gender and ALL (all three prof_gender combined)
    
    df_true = pd.read_csv(args.data_dir / f"professions_{args.eval_set}.tsv", sep="\t")
    
    profession_list = df_true['profession'].tolist()
    
    print("profession_list:", profession_list)
    
    # extract predicted distribution for current eval set
    df_pred = pd.read_csv(predicted_distribution, sep = "\t")
    
    df = pd.DataFrame()

    # extract true and predicted distribution for each gender for each profession
    for profession in profession_list:
        
        pred_dist_male = df_pred[(df_pred['profession'] == profession) & (df_pred['gender'] == 'male')]['softmax_score'].values[0]
        pred_dist_female = df_pred[(df_pred['profession'] == profession) & (df_pred['gender'] == 'female')]['softmax_score'].values[0]

        true_dist_male = df_true[df_true['profession'] == profession]['male_dist'].values[0]
        true_dist_female = df_true[df_true['profession'] == profession]['female_dist'].values[0]
        
        prof_gender = df_true[df_true['profession'] == profession]['prof_gender'].values[0]
        
        suffix = 'before' if args.model_state == 'before_debiasing' else 'after'

        temp_dict = {
            'profession': profession,
            f'f {suffix}': pred_dist_female,
            f'm {suffix}': pred_dist_male,
            'f true': true_dist_female,
            'm true': true_dist_male,
            'prof_gender': prof_gender
        }

        df_temp = pd.DataFrame([temp_dict])
    
        df = pd.concat([df, df_temp], ignore_index=True)
    
    # dataframe with individual KL (considering both male and female) for each profession 
    # dataframe with summary KL mean and variance for each prof_group and ALL
    df, df_summary = compute_kl_by_prof_gender_ALL(df)
    
    df.to_csv(BASE_PATH / f"{args.model}_individual_result_{args.model_state}_{args.eval_set}_{args.run}.tsv", sep="\t", index=False)
    df_summary.to_csv(BASE_PATH / f"{args.model}_summary_result_{args.model_state}_{args.eval_set}_{args.run}.tsv", sep="\t", index=False)

def main():
    
    gender_dict = get_attributes(args.male_gendered_words_path, args.female_gendered_words_path)    
    
    professions = get_targets(args.target_file)
    
    templates = get_templates(args.template_path)
    
    model, tokenizer = load_model(args)
        
    # invoking this method save a file with predicted distribution for male and female for each profession (NORMALIZED_BY_PROFESSION_PATH)
    get_gender_profession_distribution(model, tokenizer, templates, gender_dict, professions)
        
    get_kl_individual_and_summary(predicted_distribution = NORMALIZED_BY_PROFESSION_PATH)
    
    print("Inference completed!!")
    
if __name__ == "__main__":
    main()
