from torch.utils.tensorboard import SummaryWriter
import torch
import json
import os
import numpy as np

#from config import *

# data_path           = config.output_files
# results_file        = config.results_file
# tensorboard_folder  = config.tensorboard_folder


tensorboard_folder = "runs_tensorboards"
results_file = "results.json"
data_path = "data/outputs"

# ==============================================================================
# Tensorboards for experiments 
# ==============================================================================


# ==============================================================================
# ############### Robusted based on seeds ###############

tag = "Robustness_Seed"

for root, dirs, files in os.walk(data_path):
    
    result_json = root + f"/{results_file}"
    base_root = os.path.basename(root)

    if os.path.exists(result_json):        

        # Load data:     
        with open(f"{result_json}") as json_file:
            data = json.load(json_file)
        
        log_parameters = data["experimental_configurations"]["log_parameters"]
        seed = log_parameters["seed"]
        gradient_accumulation_steps = log_parameters["gradient_accumulation_steps"]
        
        # Sanity check to only have one event file per experiment
        if os.path.exists(f"{tensorboard_folder}/{base_root}"):
            continue

        # Get parameters to model: 
        training_loss = np.mean(np.array(data["training_loss"]).reshape(-1, gradient_accumulation_steps), axis=1)
        dev_precision = [data[z] for z in data if "dev_precision" in z]
        dev_recall = [data[z] for z in data if "dev_recall" in z]
        dev_f1 = [data[z] for z in data if "dev_f1" in z]

        # Make them tensors: 
        training_loss_tensor = torch.tensor(training_loss)
        dev_precision_tensor = torch.tensor(dev_precision[:-1])
        dev_recall_tensor = torch.tensor(dev_recall[:-1])
        dev_f1_tensor = torch.tensor(dev_f1[:-1])

        # Generate TensorBoards: 
        tb = SummaryWriter(log_dir = f"{tensorboard_folder}/{base_root}")

        # Plot loss: 
        for step, loss in enumerate(training_loss_tensor):
            tb.add_scalar(f"{tag}/00.Loss/Train", scalar_value=training_loss_tensor[step], global_step=step)
        
        for epoch, recall in enumerate(dev_f1_tensor):
            tb.add_scalar(f"{tag}/01.F1/development", scalar_value=dev_f1_tensor[epoch], global_step=epoch)
        
        for epoch, precision in enumerate(dev_precision_tensor):
            tb.add_scalar(f"{tag}/02.Precision/development", scalar_value=dev_precision_tensor[epoch], global_step=epoch)
        
        for epoch, recall in enumerate(dev_recall_tensor):
            tb.add_scalar(f"{tag}/03.Recall/development", scalar_value=dev_recall_tensor[epoch], global_step=epoch)
        
        tb.close()

# ############### Robusted based on seeds ###############
# ==============================================================================





