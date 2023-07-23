nohup python ./medclip/run_medclip.py \
    --output_dir "/mnt/eds_share/Users/yilu.zhou/Development/log/medclip/" \
    --tokenizer_name="allenai/scibert_scivocab_uncased" \
    --run_from_checkpoint="/mnt/eds_share/Users/yilu.zhou/Development/log/medclip/004" \
    --train_file="./Experiments/roco-dataset/train_dataset.json" \
    --validation_file="./Experiments/roco-dataset/validation_dataset.json" \
    --do_train --do_eval \
    --num_train_epochs 1 \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate="1e-5" --warmup_steps="0" --weight_decay 0.1 \
    --overwrite_output_dir \
    --preprocessing_num_workers 4 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    > "/mnt/eds_share/Users/yilu.zhou/Development/log/medclip/003.log" 2>&1 &
#   --push_to_hub
#   --from_pt="/mnt/eds_share/Users/yilu.zhou/Development/log/medclip/003/epoch_026.msgpack" \
#   --text_model_name_or_path="allenai/scibert_scivocab_uncased" \
#   --vision_model_name_or_path="openai/clip-vit-large-patch14-336" \

nohup python ./medclip/run_medclip.py \
    --output_dir "/mnt/eds_share/Users/yilu.zhou/Development/log/XXXX/" \
    --tokenizer_name="allenai/scibert_scivocab_uncased" \
    --run_from_checkpoint="/mnt/eds_share/Users/yilu.zhou/Development/log/XXXX/007" \
    --train_file="./Experiments/XXXX/train_dataset.json" \
    --validation_file="./Experiments/XXXX/val_dataset.json" \
    --do_train --do_eval \
    --num_train_epochs 1000 \
    --max_seq_length 32 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --overwrite_output_dir \
    --preprocessing_num_workers 4 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    > "/mnt/eds_share/Users/yilu.zhou/Development/log/XXXX/008.log" 2>&1 &