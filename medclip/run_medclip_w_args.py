import subprocess
import os

# Specify which GPU(s) to use: (number starts from 0)
# "0":   using GPU 0
# "0,1": using GPU 0 & 1
# "1":   using GPU 1
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Enter data folder name
dataset_folder_name = 'roco-dataset'
folder_temp = os.path.join('../Experiments', dataset_folder_name)

# Define the parameters for the Python script
# Make sure these values are correct for your specific use case
script_path = '../medclip/run_medclip.py'
output_dir = os.path.join('../Experiments', dataset_folder_name, 'snapshots', 'vision_augmented_biobert')
text_model_name_or_path = 'allenai/scibert_scivocab_uncased'
vision_model_name_or_path = 'openai/clip-vit-base-patch32'
tokenizer_name = 'allenai/scibert_scivocab_uncased'
train_file = os.path.join('../Experiments', dataset_folder_name, f'train_dataset.json')
validation_file = os.path.join('../Experiments', dataset_folder_name, f'validation_dataset.json')
do_train = True
do_eval = True
num_train_epochs = 40
max_seq_length = 128
per_device_train_batch_size = 32
per_device_eval_batch_size = 32
learning_rate = '1e-5'
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