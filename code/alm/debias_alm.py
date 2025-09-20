'''
Tune on DP_female, DP_male and DP_balanced (train set) with desired distribution as real-world and evaluate across all three groups (test set)

loss: weighted adpative KL Loss

Note: We focus on real world distribution as a desired distribution only as there is no bias in equality setting.
'''

import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import warnings
import functools
import random 
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
import argparse
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

access_token = "HUGGING_FACE_TOKEN"

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

print = functools.partial(print, flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
def parse_arguments():
    parser = argparse.ArgumentParser(description="Training script with hyperparameter tuning.")
    
    parser.add_argument("--seed", type=int, choices= [42, 52, 62, 72, 82], required=True, help="Random seed")
    parser.add_argument("--model", type=str, default="llama3.2_3B_Instruct", help="Name of the model (default: llama3.2_3B_Instruct).")
    parser.add_argument("--run", type=str, required=True, help="Run identifier (e.g., run1).")
    parser.add_argument("--desired_dist", type=str, choices=["equal", "real_world"], required=True, help="Desired distribution type.")
    parser.add_argument("--train_batch_size", type=int, required=True, help="Batch size for training.")
    parser.add_argument("--valid_batch_size", type=int, required=True, help="Batch size for validation.")
    parser.add_argument("--min_delta", type=float, required=True, help="Minimum delta for early stopping.")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate for optimization.")
    parser.add_argument("--WEIGHT_DECAY", type=float, required=True, help="Weight decay for regularization.")
    parser.add_argument("--patience", type=int, required=True, help="Number of epochs without improvement before stopping.")
    parser.add_argument("--momentum_weight", type=float, required=True, help="KLTracker weight")
    parser.add_argument("--batchmean", type=lambda x: x.lower() == "true", required=True, help="Use 'batchmean' reduction for KL divergence (True or False).")
        
    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=16, help="Low-rank parameter for LoRA (default: 16).")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Scaling factor for LoRA (default: 32).")
    parser.add_argument("--lora_dropout", type=float, default=0.01, help="Dropout rate for LoRA layers (default: 0.01).")
    parser.add_argument("--lora_bias", type=str, choices=["none", "all", "lora_only"], default="none", help="Bias setting for LoRA (default: none).")
    
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

def print_args(args):
    """Prints all parsed arguments in a structured format."""
    print("\n--- Parsed Arguments ---")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("-"*100)
    print()

args = parse_arguments()

make_memory_efficient = False

if args.desired_dist == 'real_world':
    df_dist = pd.read_csv("../../data/profession_dist.tsv", sep="\t")
    real_dist = df_dist.set_index('profession')[['female_dist', 'male_dist']].T.to_dict()
    
    BASE_PATH  = f"../../output_seed_{args.seed}/alm/{args.model}/{args.run}/{args.model}_real_world/tuning/"
    BASE_PATH_MODEL = f"../../debiased_llms/output_seed_{args.seed}/alm/{args.model}/{args.run}/{args.model}_real_world/tuning/best_model/"
    
ANALYSIS_PATH = BASE_PATH + "analysis/"
MODEL_PATH = BASE_PATH_MODEL
NORMALIZED_BY_PROFESSION_PATH = BASE_PATH + "dist_output/normalized_scores/"
RAW_ASSOCIATION_SCORE_PATH = BASE_PATH + "dist_output/raw_association_scores/"

MALE_ATTRIBUTE_PATH = "../../data/male_gendered_words.txt"
FEMALE_ATTRIBUTE_PATH = "../../data/female_gendered_words.txt"

TRAIN_TARGETS_PATH ="../../data/professions_train.tsv"
VALID_TARGETS_PATH ="../../data/professions_valid.tsv"

train_loss_path = os.path.join(ANALYSIS_PATH, "epoch_train_loss.pkl")
valid_loss_path = os.path.join(ANALYSIS_PATH, "epoch_valid_loss.pkl")

kl_means_path = os.path.join(ANALYSIS_PATH, "kl_means.pkl")
kl_vars_path = os.path.join(ANALYSIS_PATH, "kl_vars.pkl")
lambda_X_path = os.path.join(ANALYSIS_PATH, "lambda_X.pkl")

train_val_loss_plot_path = os.path.join(ANALYSIS_PATH, "training_vs_validation_loss.png")
kl_means_vars_plot_path = os.path.join(ANALYSIS_PATH, "kl_means_vars.png")
lambda_X_plot_path = os.path.join(ANALYSIS_PATH, "lambda_X.png")

TRAIN_TEMPLATES_PATH = "../../data/templates_train.tsv"
    
def create_folder(PATH):
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    
create_folder(BASE_PATH)
create_folder(ANALYSIS_PATH)
create_folder(MODEL_PATH)
create_folder(NORMALIZED_BY_PROFESSION_PATH)
create_folder(RAW_ASSOCIATION_SCORE_PATH)

print("-"*100)
print("Device: ", device)
print("Base path: ", BASE_PATH)
print("Model path: ", MODEL_PATH)
print("Normalized by profession path: ", NORMALIZED_BY_PROFESSION_PATH)
print("Raw association score path: ", RAW_ASSOCIATION_SCORE_PATH)
print("ANALYSIS_PATH:", ANALYSIS_PATH)
print("-"*100)
print()
print_args(args)
       
def set_seed(seed):
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed) 

    # if cuda is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed)

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

def get_targets(targets_path, dataset_split):
    '''
    dataset_split: determine if we are reading training targets or validation targets (professions)
    '''
    df_professions = pd.read_csv(targets_path, sep="\t")
       
    professions = df_professions['profession'].values.tolist() 
    prof_gender = df_professions['prof_gender'].values.tolist()
    
    print(f"Total professions ({dataset_split}):", len(professions))
    print(f"Professions({dataset_split}):", professions)
    print()
    
    return professions, prof_gender

def create_batches(professions, labels, batch_size, is_training):
    """
    Create batches of professions grouped by their labels.
    
    Parameters:
        professions (list): List of profession names.
        labels (list): Corresponding labels for each profession.
        batch_size (int): Number of items per batch.
        is_training (bool): If True, shuffles data within each label group and the batches.
    
    Returns:
        tuple: (list of batches, list of batch labels)
    """
    
    # group professions by their category
    grouped = defaultdict(list)
    for prof, lab in zip(professions, labels):
        grouped[lab].append(prof)
        
    all_batches = []
    all_batches_labels = []
    
    # For each label, shuffle the items (for training set only) and create batches
    for label, items in grouped.items():
        if is_training:
            random.shuffle(items)  # Shuffle items within the label group each epoch.
        
        # create batches using ceiling division
        num_batches = (len(items) + batch_size - 1) // batch_size  
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(items))
            batch = items[start_idx:end_idx]
            all_batches.append(batch)
            all_batches_labels.append(label)
    
    # shuffle the order of batches (for training only)
    if is_training:
        combined = list(zip(all_batches, all_batches_labels))
        random.shuffle(combined)
        all_batches, all_batches_labels = zip(*combined)
        
    return list(all_batches), list(all_batches_labels)

def compute_kl_divergence(normalized_scores, desired_dist):
    """
    Compute KL divergence and allow gradient computation with respect to the model's predicted scores.
    """
    # Define epsilon to prevent probabilities being zero
    epsilon = 1e-8

    # Target distribution (true distribution/expected distribution)
    # No need for requires_grad as this is a fixed target
    
    if desired_dist =='equal':
        target = torch.tensor([0.5, 0.5], dtype=torch.float32, requires_grad=False).to(device)
    
    kl_divergences = []

    for profession, scores in normalized_scores.items():
    
        if desired_dist == 'real_world':
            male_dist = real_dist[profession]['male_dist']
            female_dist = real_dist[profession]['female_dist']

            target = torch.tensor([male_dist, female_dist], dtype=torch.float32, requires_grad=False).to(device)

        profession_scores = torch.stack([scores['male'].to(device), scores['female'].to(device)])
        
        # Add epsilon to ensure no zero probabilities for stability
        profession_scores = torch.clamp(profession_scores, min=epsilon, max =1.0)

        # Compute log probabilities of profession scores
        log_profession_scores = torch.log(profession_scores) # use only log() instead of log_softmax() as the input is in the range of 0 to 1

        if args.batchmean:
            kl_div = F.kl_div(log_profession_scores, target, reduction='batchmean') # first param = log(predicted dist) and second param = probability of true dist
        else:
            kl_div = F.kl_div(log_profession_scores, target, reduction='sum') # first param = log(predicted dist) and second param = probability of true dist
        kl_divergences.append(kl_div)

    # Compute average KL divergence across all professions
    average_kl_div = torch.mean(torch.stack(kl_divergences))
    
    return average_kl_div

def get_normalized_by_profession(epoch, df, batch_idx, dataset_split):
    # Sum exponentials by profession and gender
    exp_sums = df.groupby(['profession', 'gender'])['exp_score'].sum().reset_index()

    # Compute total exponentials sum per profession
    exp_sums['total_exp'] = exp_sums.groupby('profession')['exp_score'].transform('sum')

    # Normalize by dividing gender-specific exponential sum by the total per profession
    exp_sums['softmax_score'] = exp_sums['exp_score'] / exp_sums['total_exp']

    df_normalized_by_profession = exp_sums[['profession', 'gender', 'softmax_score']]
    
    create_folder(NORMALIZED_BY_PROFESSION_PATH + dataset_split)
        
    df_normalized_by_profession.to_csv(NORMALIZED_BY_PROFESSION_PATH + dataset_split + f"normalized_gender_prof_dist_epoch_{epoch+1}_batch_{batch_idx+1}.tsv", sep = "\t", index=False)
    
def compute_normalized_scores_by_profession(norm_scores):
    """
    For each profession, compute normalized score for male and female such that the sum of the two is 1.
    Here we use the softmax function to normalize the scores.
    
    norm_scores: Dictionary of professions containing lists of raw scores for 'male' and 'female'
    Example format: {'doctor': {'male': [-1.1, -2, -3], 'female': [1, -1.5, -3]}}
    
    male values are all the male gendered scores for a profession across templates
    """
    normalized_scores = {}
    
    for profession, scores in norm_scores.items():
        
        male_scores = torch.exp(-torch.stack(scores['male']))  # Shape: [N] using torch.stack does not require grad to be set to True as it is implicitly set to True
        female_scores = torch.exp(-torch.stack(scores['female']))  # Shape: [N] using torch.stack does not require grad to be set to True as it is implicitly set to True
        
        # sum the exponentials for each gender
        sum_male = torch.sum(male_scores)
        sum_female = torch.sum(female_scores)
        
        # Calculate the total sum of all exponentials
        total_sum = sum_male + sum_female
        
        # Normalize each gender's sum by dividing by the total sum
        male_normalized_score = sum_male / total_sum
        female_normalized_score = sum_female / total_sum
        
        normalized_scores[profession] = {'male': male_normalized_score, 'female': female_normalized_score}

    return normalized_scores

def calculate_loss_and_perplexity(model, tokenizer, sentence):
    
    if args.model == 'llama3.2_3B_Instruct':
        model.to(device)

    if args.model == 'llama3.2_3B_Instruct':
        tokens = tokenizer(sentence, return_tensors='pt').to(device)
    else:
        tokens = tokenizer(sentence, return_tensors='pt')

    outputs = model(**tokens, labels=tokens['input_ids'])
    
    loss = outputs.loss  # keep as tensor
    perplexity = torch.exp(loss)  # still a tensor, retains graph

    return loss, perplexity

def get_gender_profession_distribution(epoch, model, tokenizer, templates,gender_dict, professions, batch_idx, dataset_split):
    
    df = pd.DataFrame()
    
    association_scores  = {profession: {'male': [], 'female': []} for profession in professions}
    
    for template_id, template in templates.items():
        
        template_actual = template
        
        for target in professions:
            
            # print(f"Template: '{template_id}', profession: {target}")
            
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
                            'association_score': sentence_loss.item(),
                            'perplexity': perplexity.item()
                            }
                
                association_scores[target][gender].append(sentence_loss)
                
                df_temp = pd.DataFrame([temp_dict])
                df = pd.concat([df, df_temp], ignore_index=True)
    
    #####################################################################
    '''
    This block saves the intermediate gender-profession distribution during finetuning for each epoch and batch.
    '''
    # Compute exponentials of the scores
    df['exp_score'] = np.exp(-df['association_score'])
    
    create_folder(RAW_ASSOCIATION_SCORE_PATH + dataset_split)
        
    df.to_csv(RAW_ASSOCIATION_SCORE_PATH + dataset_split + f"raw_association_epoch_{epoch+1}_batch_{batch_idx+1}.tsv", sep = "\t", index=False) 
    
    # save normalized male and female score for each profession a tsv file
    get_normalized_by_profession(epoch, df, batch_idx, dataset_split)
    #####################################################################
    
    normalized_scores  = compute_normalized_scores_by_profession(association_scores)
    
    return normalized_scores

def load_model():
    print("Loading tokenizer and model...")
    
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
        
    # llama3.2_3B_Instruct: non-quantized, apply LoRA
    if args.model == 'llama3.2_3B_Instruct':
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_auth_token=access_token)
        model = AutoModelForCausalLM.from_pretrained(args.model_id, 
                                                     use_auth_token=access_token, 
                                                     #  device_map="auto",  # spreads model across available GPUs/CPU
                                                     torch_dtype=torch.float16)

        print("Max seq length:", tokenizer.model_max_length)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        '''
        Attention: q_proj, k_proj, v_proj, o_proj

        Feedforward (MLP): gate_proj, up_proj, down_proj
        '''
        # QLoRA configuration
        lora_config = LoraConfig(
            r=args.lora_r, 
            lora_alpha=args.lora_alpha,  
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,     
            bias=args.lora_bias,  
            task_type="CAUSAL_LM",
            inference_mode=False,  # Set to True for inference mode, False for training (while loading saved model, it is automatically set to True)
        )
    
        model = get_peft_model(model, lora_config)

        # training mode flag
        model.config.use_cache = False
        
        model.print_trainable_parameters()

        model.to(device)
        
    # Quantized + QLoRA for other models
    elif args.model in quantized_models: 
        quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16)

        trust_code = args.model in qwen_models
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, 
                                                  trust_remote_code=trust_code,
                                                  use_auth_token=None if trust_code else access_token
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(args.model_id,
                                                     trust_remote_code=trust_code,
                                                     use_auth_token=None if trust_code else access_token,
                                                     quantization_config=quant_config,
                                                     device_map="auto")
        
        # prepare QLoRA model for finetuning
        model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type="CAUSAL_LM",
            inference_mode=False,
        )
            
        model = get_peft_model(model, lora_config)
        
        # training mode flag
        model.config.use_cache = False
    
    
    print(f"LoRA Model Initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    return model, tokenizer


def save_plot(x, y_values, labels, title, xlabel, ylabel, filename):
    """
    Saves a plot with multiple lines representing different y-values.

    Parameters:
    - x: List of x-axis values (epochs).
    - y_values: List of lists, each containing y-axis values for a different line.
    - labels: List of labels corresponding to each line.
    - title: Plot title.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - filename: Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    for y, label in zip(y_values, labels):
        plt.plot(x, y, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_training_validation_loss(epoch_train_loss, epoch_valid_loss, filename="training_vs_validation_loss.png"):
    """
    Plots training vs validation loss across epochs.
    """
    epochs = list(range(1, len(epoch_train_loss) + 1))
    save_plot(
        x=epochs,
        y_values=[epoch_train_loss, epoch_valid_loss],
        labels=["Training Loss", "Validation Loss"],
        title="Training vs Validation Loss",
        xlabel="Epoch",
        ylabel="Loss",
        filename=filename
    )

def plot_kl_means_vars(kl_means, kl_vars, filename="kl_means_vars.png"):
    """
    Plots KL means and variances for each category across epochs.
    """
    categories = kl_means.keys()
    epochs = list(range(1, len(next(iter(kl_means.values()))) + 1))

    kl_means_values = [kl_means[cat] for cat in categories]
    kl_vars_values = [kl_vars[cat] for cat in categories]

    labels_means = [f"KL Mean ({cat})" for cat in categories]
    labels_vars = [f"KL Variance ({cat})" for cat in categories]

    save_plot(
        x=epochs,
        y_values=kl_means_values + kl_vars_values,
        labels=labels_means + labels_vars,
        title="KL Means and Variances Across Epochs",
        xlabel="Epoch",
        ylabel="KL Value",
        filename=filename
    )

def plot_lambda_X(lambda_X_values, filename="lambda_X.png"):
    """
    Plots lambda_X values across epochs.
    """
    
    # If lambda_X_values is a PyTorch tensor, move it to CPU and convert to a numpy array
    if isinstance(lambda_X_values, torch.Tensor):
        lambda_X_values = lambda_X_values.detach().cpu().numpy()
        
    # Ensure it's a flat numpy array
    lambda_X_values = np.asarray(lambda_X_values).flatten()
    
    epochs = list(range(1, len(lambda_X_values) + 1))
    
    save_plot(
        x=epochs,
        y_values=[lambda_X_values],
        labels=["Lambda_X"],
        title="Lambda_X Across Epochs",
        xlabel="Epoch",
        ylabel="Lambda_X",
        filename=filename
    )
class KLTracker:
    """
    Tracks the moving KL mean and variance for each category over multiple batches.
    Uses Exponential Moving Average (EMA) for mean and Welford's method for variance.
    """
    def __init__(self, beta=0.95): 
        self.beta = beta
        self.kl_means = {}  # Moving average KL loss per category
        self.kl_vars = {}  # KL variance per category
        self.kl_counts = {}  # Track occurrences of each category
        
        # New: Store historical values per epoch
        self.kl_means_history = {}  # Stores KL mean per category across epochs
        self.kl_vars_history = {}  # Stores KL variance per category across epochs

    def update(self, category, new_kl, batch_size):
        """
        Updates the moving KL mean and variance for the given category.
        Even if only one category is updated at a time, previous statistics are retained.
        """
        if category not in self.kl_means:
            # First-time initialization for the category
            self.kl_means[category] = new_kl.clone()  # Store tensor, not Python float
            self.kl_vars[category] = torch.tensor(0.0, device=new_kl.device, dtype=new_kl.dtype)
            self.kl_counts[category] = torch.tensor(batch_size, device=new_kl.device, dtype=new_kl.dtype)
            
            # Initialize history tracking
            self.kl_means_history[category] = []
            self.kl_vars_history[category] = []
        else:
            # Save the old mean before updating
            old_mean = self.kl_means[category]
            old_var = self.kl_vars[category]
            previous_count = self.kl_counts[category]
            
            # Update the count by the current batch size
            self.kl_counts[category] += batch_size
            total_count = self.kl_counts[category]

            # Update the moving mean using EMA
            new_mean = self.beta * old_mean + (1 - self.beta) * new_kl
            self.kl_means[category] = new_mean  # Preserve tensor computation

            # Update variance using Welford’s method (compatible with EMA)(ensure tensor operations)
            self.kl_vars[category] = (previous_count * old_var + (new_kl - new_mean) * (new_kl - old_mean)) / total_count
           
        return self.kl_means[category], self.kl_vars[category]
    
    def store_epoch_values(self):
        """
        Stores the KL mean & variance for all categories at the end of an epoch.
        This is to track values over multiple epochs.
        """
        for category in self.kl_means.keys():
            self.kl_means_history[category].append(self.kl_means[category].item())
            self.kl_vars_history[category].append(self.kl_vars[category].item())

def adjust_lambda(category, kl_means, kl_vars):
    """
    Adjusts lambda_X dynamically based on both KL mean and variance.
    
    Controls how fast the updates should be applied? (how aggressive we change model weight)
    - If variance is high -> most unstable -> slower update
    - If variance is low -> most stable -> faster update
    """
    # torch.clamp(x, 1.1,1.5) (x, UB, LB) equivalent to max(min(x, 1.5), 1.1)
    # torch.clamp() preserve computation graph
    
    kl_var_factor = 1 / (1 + kl_vars[category])

    if category in ["male_dominated", "female_dominated"]:
        lambda_X = torch.clamp(0.95 * kl_means[category] * kl_var_factor, 0.8, 1.5)

    elif category in ["balanced"]:
        lambda_X = torch.clamp(1.2 * kl_means[category] * kl_var_factor, 1.0, 1.5)
    
    return lambda_X

def main():
    
    set_seed(args.seed)
    
    templates = get_templates(template_path=TRAIN_TEMPLATES_PATH)
    
    gender_dict = get_attributes(male_attribute_path= MALE_ATTRIBUTE_PATH, female_attribute_path= FEMALE_ATTRIBUTE_PATH)    
    
    professions_train, prof_gender_train = get_targets(targets_path=TRAIN_TARGETS_PATH, dataset_split='train')
    professions_valid, prof_gender_valid = get_targets(targets_path=VALID_TARGETS_PATH, dataset_split='valid')
    valid_batches, valid_batches_prof_gender = create_batches(professions_valid, prof_gender_valid, batch_size=args.valid_batch_size, is_training=False)
    
    model, tokenizer = load_model()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay=args.WEIGHT_DECAY) # L2 regularization using weight decay
    
    # Show trainable parameters (LoRA layers)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable Parameter: {name} | Shape: {param.shape}")
    
    epoch = 0
    best_val_loss = float('inf')  # track the best validation loss
    num_increases = 0  # Counter for epochs without improvement
    
    epoch_train_loss = [] # track training loss
    epoch_valid_loss = []
    epoch_lambda_X_values = []   # Stores lambda_X values per epoch
    
    kl_tracker = KLTracker(beta=args.momentum_weight)
    
    while True:
        train_batches, train_batches_prof_gender = create_batches(professions_train, prof_gender_train, batch_size=args.train_batch_size, is_training=True)
        
        # set model.train() before every epoch to ensure that the model is in training mode and create computation graph
        model.train()
        
        total_loss = 0
        count = 0 # keep track of total professions (total training instances)
        
        for batch_idx, (profession_batch, batch_category) in enumerate(zip(train_batches, train_batches_prof_gender)):
                                   
            total_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

            print(f"Epoch {epoch + 1}, Processing batch {batch_idx+1}/{len(train_batches)}...") 
            print(f"Total trainable parameters: {total_params}")
                        
            # use mixed precision
            with autocast(enabled=make_memory_efficient):
                
                # normalized_scores: normalized male and female scores for each profession
                normalized_scores = get_gender_profession_distribution(epoch, model, tokenizer, templates, gender_dict,profession_batch, batch_idx, "train/")
                
                loss = compute_kl_divergence(normalized_scores, args.desired_dist)        
            
            print("Batch category:", batch_category, profession_batch)
            
            if args.desired_dist == 'real_world':
                
                # Update moving KL mean & variance dynamically
                kl_mean, kl_var = kl_tracker.update(batch_category, loss.detach(), len(profession_batch)) # avoid loss.item() to preserve computation graph
                    
                if batch_category in ["male_dominated", "female_dominated"]:
                    alpha = 1e-6  
                
                elif batch_category in ["balanced"]: 
                    alpha = 1e-5  
                
                normalized_loss = loss / (kl_mean + alpha) # alpha (float) is promoted to a tensor due to broadcasting
                                
                # Dynamically adjust lambda_X based on past category dominance
                lambda_X = adjust_lambda(batch_category, kl_tracker.kl_means, kl_tracker.kl_vars) 
               
                # Apply weighted kl loss
                weighted_loss = lambda_X * normalized_loss
                
            # use weighted loss
            if args.desired_dist == 'real_world':
                # Ensure loss is already a PyTorch tensor and potentially already on the correct device (GPU)
                if not isinstance(weighted_loss, torch.Tensor):
                    weighted_loss = torch.tensor(weighted_loss, requires_grad=True)
                    # raise ValueError("weighted_loss should be a tensor computed from a differentiable operation.")
                
                # move the loss to the GPU (if the model is in CUDA)
                weighted_loss = weighted_loss.to(model.device)        
                
                # Check if loss requires grad
                if not weighted_loss.requires_grad:
                    raise RuntimeError("Loss does not require grad. Check your loss computation.")
            
            optimizer.zero_grad() # zero out gradients before backpropagation
            
            if args.desired_dist == 'real_world':
                weighted_loss.backward() # backpropagate the loss; retain_graph=True for visualizing computation graph
                    
            optimizer.step() # update model parameters    
            
            '''
            The product of weighted_loss.item() and len(profession_batch) gives the total loss 
            for the current batch scaled by the batch size, 
            ensuring that each batch contributes proportionately to the total loss, 
            especially important when the last batch may be smaller than the others
            if the total number of professions isn't divisible by the batch size.
            '''
            if args.desired_dist == 'real_world':
                total_loss += weighted_loss.item() * len(profession_batch)
            
            '''
            This accumulates the total number of profession entries processed so far
            across all batches in the current epoch. 
            It sums up the sizes of all the batches processed, 
            which is used later to compute the average loss.
            '''
            count += len(profession_batch)
               
        if args.desired_dist == 'real_world':
            # Store KL Mean and Variance values at the end of each epoch
            kl_tracker.store_epoch_values()
            epoch_lambda_X_values.append(lambda_X)

            print(f"KL Mean Epoch {epoch + 1}: {kl_tracker.kl_means}")
            print(f"KL Variance Factor Epoch {epoch + 1}: {kl_tracker.kl_vars}")
            print(f"Lambda_X: {lambda_X}")   
            print()
        
        # Compute average training loss for the epoch
        average_loss = total_loss / count
        epoch_train_loss.append(average_loss)
            
        print(f"Completed Epoch {epoch + 1}: Training Loss: {average_loss}")
        
        # ======================= VALIDATION LOSS COMPUTATION =======================
        total_valid_loss = 0
        valid_count = 0
        
        model.eval() # Disables dropout/batchnorm updates
        with torch.no_grad(): # Disable gradient computation during validation
            for valid_batch_idx, (valid_profession_batch, valid_batch_category) in enumerate(zip(valid_batches, valid_batches_prof_gender)):
                
                with autocast(enabled=make_memory_efficient):
                    # for validation we only focus on KL loss (no need of weight) as we use weight for training only
                    # during inference on validaiton and testing set we only focus on KL loss (without weight)
                    normalized_scores_valid  = get_gender_profession_distribution(epoch, model, tokenizer, templates, gender_dict, valid_profession_batch, valid_batch_idx, "valid/")
                    
                    # we only evalaute KL loss (unweighted)
                    valid_loss = compute_kl_divergence(normalized_scores_valid, args.desired_dist)
                
                total_valid_loss += valid_loss.item() * len(valid_profession_batch)
                
                valid_count += len(valid_profession_batch)
            
            # Compute average validation loss
            average_valid_loss = total_valid_loss / valid_count
            epoch_valid_loss.append(average_valid_loss)
                    
            print(f"Completed Epoch {epoch + 1}: Validation Loss: {average_valid_loss}")
            
        print("-"*100)
        
        # ======================== EARLY STOPPING BASED ON VALIDATION LOSS ========================
        if best_val_loss - average_valid_loss >= args.min_delta: # min_delta to prevent updating weights when improvement is too small
            print(f"New Best Validation Loss: {average_valid_loss:.6f} at Epoch {epoch+1}. Saving Model.")
            best_val_loss = average_valid_loss # Update the best loss
            num_increases = 0  # Reset early stopping counter
            model.save_pretrained(MODEL_PATH)
            tokenizer.save_pretrained(MODEL_PATH)
        else:
            num_increases += 1
            print(f"Validation Loss Did Not Improve for {num_increases} Consecutive Epochs.")
        if num_increases >= args.patience:
            print(f"Early Stopping at Epoch {epoch + 1} After {args.patience} Epochs Without Improvement.")
            break
                
        epoch += 1
            
    # ========== Save losses ==========
    with open(train_loss_path, 'wb') as f:
        pickle.dump(epoch_train_loss, f)
        
    with open(valid_loss_path, 'wb') as f:
        pickle.dump(epoch_valid_loss, f)

    if args.desired_dist == 'real_world':
        # Save KL means and variances
        with open(kl_means_path, 'wb') as f:
            pickle.dump(kl_tracker.kl_means_history, f)
            
        with open(kl_vars_path, 'wb') as f:
            pickle.dump(kl_tracker.kl_vars_history, f)
        
        # Save lambda_X values
        with open(lambda_X_path, 'wb') as f:
            pickle.dump(epoch_lambda_X_values, f)

    # ========== Plotting ==========
    plot_training_validation_loss(epoch_train_loss, epoch_valid_loss, train_val_loss_plot_path)
    
    if args.desired_dist == 'real_world':
        plot_kl_means_vars(kl_tracker.kl_means_history, kl_tracker.kl_vars_history, kl_means_vars_plot_path)
        plot_lambda_X(epoch_lambda_X_values, lambda_X_plot_path)
    
    print("\nFine-tuning completed!!!")
    
if __name__ =="__main__":
    main()
