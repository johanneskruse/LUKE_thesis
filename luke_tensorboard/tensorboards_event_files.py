from torch.utils.tensorboard import SummaryWriter
import torch
import json
import os
import numpy as np

from argparse import Namespace
import click

# ==============================================================================
# Tensorboards for experiments 
# ==============================================================================

# ==============================================================================
# ############### Robusted based experiment_tags ###############

class config():
    tags = ["learning_rate", "seed", "train_batch_size", "train_frac_size"]


@click.command()
@click.option("--data-dir", default="data/outputs/OpenEntity", type=click.Path(exists=True))
@click.option("--tensorboard-event-folder", default="luke_tensorboard/runs_tensorboards")
def run(**task_args):
    args = Namespace(**task_args)

    data_dir = args.data_dir
    tensorboard_event_folder = args.tensorboard_event_folder

    # Tag used for the tensorboard: 
    experiment_tags = config.tags
    print(data_dir)

    for experiment_tag in experiment_tags:

        for root, dirs, files in os.walk(data_dir):

            result_json = root + "/results.json"
            base_root = os.path.basename(root)
            
            if os.path.exists(result_json) and experiment_tag in base_root:
                
                # Load data:     
                with open(f"{result_json}") as json_file:
                    data = json.load(json_file)
                
                log_parameters = data["experimental_configurations"]["log_parameters"]
                par = log_parameters[f"{experiment_tag}"]
                gradient_accumulation_steps = log_parameters["gradient_accumulation_steps"]
                
                # Sanity check to only have one event file per experiment
                if os.path.exists(f"{tensorboard_event_folder}/{base_root}"):
                    continue

                # Get parameters to model: 

                if len(data["training_loss"]) % gradient_accumulation_steps != 0: 
                    print(f"Odd number of global_steps and training_loss: {base_root}")
                    data["training_loss"].append(data["training_loss"][-1])
                    training_loss = np.mean(np.array(data["training_loss"]).reshape(-1, gradient_accumulation_steps), axis=1)
                else: 
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
                tb = SummaryWriter(log_dir = f"{tensorboard_event_folder}/{base_root}")

                # Plot loss: 
                for step, loss in enumerate(training_loss_tensor):
                    tb.add_scalar(f"{experiment_tag}/00.Loss/Train", scalar_value=training_loss_tensor[step], global_step=step)
                
                for epoch, recall in enumerate(dev_f1_tensor):
                    tb.add_scalar(f"{experiment_tag}/01.F1/development", scalar_value=dev_f1_tensor[epoch], global_step=epoch)
                
                for epoch, precision in enumerate(dev_precision_tensor):
                    tb.add_scalar(f"{experiment_tag}/02.Precision/development", scalar_value=dev_precision_tensor[epoch], global_step=epoch)
                
                for epoch, recall in enumerate(dev_recall_tensor):
                    tb.add_scalar(f"{experiment_tag}/03.Recall/development", scalar_value=dev_recall_tensor[epoch], global_step=epoch)
                
                tb.close()


# ############### Robusted based on seeds ###############
# ==============================================================================


if __name__ == '__main__':
    run()