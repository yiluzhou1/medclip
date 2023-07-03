import subprocess

# Define the parameters for the Python script
# Make sure these values are correct for your specific use case
script_path = './medclip/run_medclip.py'
output_dir = './snapshots/vision_augmented_biobert'
text_model_name_or_path = 'allenai/scibert_scivocab_uncased'
vision_model_name_or_path = 'openai/clip-vit-base-patch32'
tokenizer_name = 'allenai/scibert_scivocab_uncased'
train_file = 'data/train_dataset_new.json'
validation_file = 'data/valid_dataset_new.json'
do_train = True
do_eval = True
num_train_epochs = 40
max_seq_length = 128
per_device_train_batch_size = 16
per_device_eval_batch_size = 16
learning_rate = '5e-5'
warmup_steps = 0
weight_decay = 0.1
overwrite_output_dir = True
preprocessing_num_workers = 4

# Run the Python script with the parameters
command = ['python', script_path,
           '--output_dir', output_dir,
           '--text_model_name_or_path', text_model_name_or_path,
           '--vision_model_name_or_path', vision_model_name_or_path,
           '--tokenizer_name', tokenizer_name,
           '--train_file', train_file,
           '--validation_file', validation_file,
           # '--do_train', '--do_eval',
           '--num_train_epochs', str(num_train_epochs),
           '--max_seq_length', str(max_seq_length),
           '--per_device_train_batch_size', str(per_device_train_batch_size),
           '--per_device_eval_batch_size', str(per_device_eval_batch_size),
           '--learning_rate', learning_rate,
           '--warmup_steps', str(warmup_steps),
           '--weight_decay', str(weight_decay),
           # '--overwrite_output_dir',
           '--preprocessing_num_workers', str(preprocessing_num_workers),
          ]

if do_train:
    command.append('--do_train')

if do_eval:
    command.append('--do_eval')

if overwrite_output_dir:
    command.append('--overwrite_output_dir')

subprocess.run(command)