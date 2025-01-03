
deepspeed --num_gpus=2 ./src/evaluation/deepspeed_evaluation.py \
    --data_path ../datas/test1 \
    --task_type all \
    --data_type train \
    --model_name Qwen/Qwen2-VL-7B-Instruct \
    --cache_dir /gemini/pretrain/hub \
    --ckpt_save_path ./checkpoints/ \
    --load_file_name best.ckpt \
    --batch_size 1 \
    --num_workers 2 \
    --mixed_precision False \
    --wrapper_type transformer \
    --sharding_strategy fsdp \
    --low_cpu_fsdp \
    --fsdp_activation_checkpointing \
    --selective_checkpointing 1/3 \
    --use_torch_compile True\
    --use_profiler \
    --profile_traces ./logs/profiler \
    --use_lora \
    --low_rank 8 \
    --target_modules q_proj v_proj \
    --peft_type lora \
    --lora_alpha 8 \
    --bias none \
    --adapter_name sft_qwen_vl \
    --shuffle True\
    --seed 567\
    --tracker wandb\
    --tracker_dir ./logs/tracker\
    --tracker_project_name mllm\
    --profiler_rank0_only True\
    --sft False