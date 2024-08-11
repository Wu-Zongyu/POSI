# POSI
Code for our paper "Universal Prompt Optimizer for Safe Text-to-Image Generation"

## SFT Training
```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --model_name_or_path huggyllama/llama-7b \
    --do_train True \
    --finetuning_type lora \
    --template default \
    --flash_attn False \
    --shift_attn False \
    --dataset_dir data \
    --dataset sft_toxic_clean \
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --neft_alpha 0 \
    --train_on_prompt False \
    --upcast_layernorm False \
    --lora_rank 8 \
    --lora_dropout 0.1 \
    --lora_target q_proj,v_proj \
    --resume_lora_training True \
    --output_dir saves/LLaMA-7B/lora/SFT \
    --fp16 True \
    --plot_loss True
```
## PPO Training
For PPO training, a reward model needs to be trained as a place holder first.

## Acknowledgement
Our code is based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Thanks for their fantastic work!
