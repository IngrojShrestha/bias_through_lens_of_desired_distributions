# LLM Bias Detection and Mitigation through the Lens of Desired Distributions
*Accepted at EMNLP 2025*

## MLM Bias Detection and Mitigation

#### Bias Detection Before Debiasing

We do not need to set seed (or set 42 to avoid error) for bias detection before debiasing.

```
python3 code/mlm/mlm_bias_detection.py \
        --seed 42 \
        --model bert_base or distilbert or bert_large \
        --batchmean True \
        --run run_name \
        --eval_set test or valid \
        --desired_dist equal or real_world \
        --model_state before_debiasing \
        --base_output_path ../../output_seed_42/mlm
```

#### Bias Mitigation

Equal Distribution

*Non-adaptive KL Loss*

```
python3 code/mlm/debias_mlm.py \
        --seed 42 or 52 or 62 or 72 or 82 \
        --model bert_base or distilbert or bert_large \
        --run run_name \
        --desired_dist equal \
        --train_batch_size 8 or 5 \
        --valid_batch_size 3 \
        --min_delta 0.0001 \
        --lr 2e-5 \
        --WEIGHT_DECAY 0.01 \
        --patience 5 \
        --momentum_weight 1.0 \
        --batchmean True \
```

Real World Distribution

*Weighted adaptive KL Loss + MLM loss*

```
python3 code/mlm/debias_mlm.py \
        --seed 42 or 52 or 62 or 72 or 82 \
        --model bert_base or distilbert or bert_large \
        --run run_name \
        --desired_dist real_world \
        --train_batch_size 8 or 5 \
        --valid_batch_size 3 \
        --min_delta 0.001 \
        --lr 2e-5 \
        --WEIGHT_DECAY 0.01 \
        --patience 5 \
        --momentum_weight 0.60 or 0.80 or 0.95 \
        --batchmean True \
        --gamma 0.001 or 0.01 or 0.1 or 0.2 or 0.5 or 0.8 or 1.0 \
```

#### Bias Detection After Debiasing (Equal Distribution)

```
python3 code/mlm/mlm_bias_detection.py \
        --seed 42 or 52 or 62 or 72 or 82 \
        --model bert_base or distilbert or bert_large \
        --batchmean True \
        --run run_name \
        --eval_set test or valid \
        --desired_dist equal \
        --model_state after_debiasing \
        --base_output_path ../../output_seed_{seed}/mlm
```

#### Bias Detection After Debiasing (Real world Distribution)

```
python3 code/mlm/mlm_bias_detection.py \
        --seed 42 or 52 or 62 or 72 or 82 \
        --model bert_base or distilbert or bert_large \
        --batchmean True \
        --run run_name \
        --eval_set test or valid \
        --desired_dist real_world \
        --model_state after_debiasing \
        --base_output_path ../../output/mlm \
        --gamma 0.001 or 0.01 or 0.1 or 0.2 or 0.5 or 0.8 or 1.0
```

#### Evaluate MLM loss

```
python3 code/mlm/evaluate_mlm_loss.py \
        --seed 42 or 52 or 62 or 72 or 82 \
        --model bert_base or distilbert or bert_large \
        --run run_name \
        --desired_dist equal or real_world \
        --model_state before_debiasing or after_debiasing \
        --base_output_path ../../output_seed_{seed}/mlm
```

#### Evaluate GLUE performance

We run [GLUE](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) from transformers.


## Autoregressive LLM (ALMs) Bias Detection and Mitigation

#### Bias Detection Before Debiasing

We do not need to set seed (or set 42 to avoid error) for bias detection before debiasing.

```
python3 code/alm/alm_bias_detection.py \
        --seed 42 \
        --model llama3.2_3B_Instruct or llama3.1_8B_Instruct or llama3.3_70B_Instruct or qwen2.5_7B_Instruct or qwen2.5_72B_Instruct \
        --batchmean True \
        --run run_name \
        --eval_set test or valid \
        --desired_dist equal or real_world \
        --model_state before_debiasing \
        --base_output_path ../../output_seed_42/alm
```

#### Bias Mitigation

*Weighted adaptive KL Loss*

```
python3 code/alm/debias_alm.py \
        --seed 42 or 52 or 62 or 72 or 82 \
        --model llama3.1_8B_Instruct or llama3.3_3B_Instruct \
        --run run_name \
        --desired_dist real_world \
        --train_batch_size 8 or 5 \
        --valid_batch_size 3 \
        --min_delta 0.001 \
        --lr 2e-5 or 2e-4 \
        --WEIGHT_DECAY 0.01 \
        --patience 5 \
        --momentum_weight 0.60 or 0.80 or 0.95 \
        --batchmean True \
        --lora_r 64 \
        --lora_alpha 16 or 32 or 64 \
        --lora_dropout 0.2 \
        --lora_bias none \
```

#### Bias Detection After Debiasing (Real World Distribution)

```
python3 code/alm/alm_bias_detection.py \
        --seed 42 or 52 or 62 or 72 or 82 \
        --model llama3.2_3B_Instruct or llama3.1_8B_Instruct \
        --batchmean True \
        --run run_name \
        --eval_set test or valid \
        --desired_dist real_world \
        --model_state after_debiasing \
        --base_output_path ../../output_seed_{seed}/alm
```

#### Evaluate Perplexity on external corpus

Before Debiasing

We do not need to set seed (or set 42 to avoid error) for evaluating perplexity on external corpus before debiasing.

```
python3 code/alm/evaluate_perplexity_external_dataset.py \
        --seed 42 \
        --model_name llama3.2_3B_Instruct or llama3.1_8B_Instruct or llama3.3_70B_Instruct or qwen2.5_7B_Instruct or qwen2.5_72B_Instruct \
        --run run_name \
        --desired_dist equal or real_world \
        --model_state before_debiasing \
        --base_output_path ../../output_seed_42/alm
```

After Debiasing

```
python3 code/alm/evaluate_perplexity_external_dataset.py \
        --seed 42 or 52 or 62 or 72 or 82 \
        --model llama3.2_3B_Instruct or llama3.1_8B_Instruct \
        --run run_name \
        --desired_dist real_world \
        --model_state after_debiasing \
        --base_output_path ../../output_seed_{seed}/alm
```

#### LM Evaluation Harness
[LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness/)

Evaluation on five benchmarks: HellaSwag, LAMBADA (OpenAI), TruthfulQA (generation), MMLU, and GLUE