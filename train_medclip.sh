python ./medclip/run_medclip.py \
    --output_dir "/mnt/eds_share/Users/yilu.zhou/Development/log/medclip/" \
    --text_model_name_or_path="allenai/scibert_scivocab_uncased" \
    --vision_model_name_or_path="openai/clip-vit-large-patch14-336" \
    --tokenizer_name="allenai/scibert_scivocab_uncased" \
    --train_file="./Experiments/roco-dataset/train_dataset.json" \
    --validation_file="./Experiments/roco-dataset/validation_dataset.json" \
    --do_train --do_eval \
    --num_train_epochs= 1 --max_seq_length 128 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --learning_rate="1e-6" --warmup_steps="0" --weight_decay 0.1 \
    --overwrite_output_dir \
    --preprocessing_num_workers 4
#    --push_to_hub
