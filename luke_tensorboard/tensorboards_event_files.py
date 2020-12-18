from torch.utils.tensorboard import SummaryWriter
import torch
import json
import os
import numpy as np

from argparse import Namespace
import click

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
plt.rc('font', size=18)
plt.rc('axes', titlesize=20)

# ==============================================================================
# Tensorboards for experiments 
# ==============================================================================

class config():
    tags = ["learning_rate", "seed", "train_batch_size", "train_frac_size"]

# ==============================================================================
# ############### Robusted based experiment_tags ###############

# ==============================================================================
# Tensorboards for experiments 
# ==============================================================================

def plot_scatter(eval_dict, title, labels=["Development", "Test"]):
            
    figure = plt.figure(figsize=(10,7))
                
    max_min = []
    for tag in eval_dict:
        dev = eval_dict[tag][:,0]
        test = eval_dict[tag][:,1]

        plt.scatter(dev, test, label=f"{tag}")

        max_min.extend([min(dev), min(test), max(dev), max(test)])

    dummy = np.linspace(min(max_min)-0.01,  max(max_min)+0.01, 2)
    plt.plot(dummy, dummy)
                
    plt.title(title, fontsize=24)
    plt.grid()
    plt.legend()
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.tight_layout()
    return figure


@click.command()
@click.option("--data-dir", default="data/outputs/OpenEntity", type=click.Path(exists=True))
@click.option("--output-dir", default="luke_tensorboard")
@click.option("--scatter-plot/--no-scatter-plot", default=True)
def run(**task_args):
    args = Namespace(**task_args)

    data_dir = args.data_dir    
    output_dir = args.output_dir
    
    tensorboard_event_folder = os.path.join(args.output_dir, "runs_tensorboards") 

    # Tag used for the tensorboard: 
    experiment_tags = config.tags

    ####### COPY START #######
    eval_results = {}

    for experiment_tag in experiment_tags:

        f1_eval = torch.tensor([])
        precision_eval = torch.tensor([])
        recall_eval = torch.tensor([])
        
        for root, dirs, files in os.walk(data_dir):

            result_json = os.path.join(root, "results.json")
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

                # Training: 
                training_loss   = torch.tensor(training_loss)
                dev_f1          = torch.tensor([data[z] for z in data if "dev_f1_epoch" in z])
                dev_precision   = torch.tensor([data[z] for z in data if "dev_precision_epoch" in z])
                dev_recall      = torch.tensor([data[z] for z in data if "dev_recall_epoch" in z])
    
                # Evaluation
                dev_f1_eval         = torch.tensor([data[z] for z in data if "dev_f1" in z and "_epoch" not in z])
                test_f1_eval        = torch.tensor([data[z] for z in data if "test_f1" in z])
                
                dev_precision_eval  = torch.tensor([data[z] for z in data if "dev_precision" in z and "_epoch" not in z])
                test_precision_eval = torch.tensor([data[z] for z in data if "test_precision" in z])
                
                dev_recall_eval     = torch.tensor([data[z] for z in data if "dev_recall" in z and "_epoch" not in z])
                test_recall_eval    = torch.tensor([data[z] for z in data if "test_recall" in z])

                f1_eval = np.append(f1_eval, [dev_f1_eval, test_f1_eval])
                precision_eval = np.append(precision_eval, [dev_precision_eval, test_precision_eval])
                recall_eval = np.append(recall_eval, [dev_recall_eval, test_recall_eval])

                # Generate TensorBoards: 
                tb = SummaryWriter(log_dir = f"{tensorboard_event_folder}/{base_root}")

                # Plot loss: 
                for step, loss in enumerate(training_loss):
                    tb.add_scalar(f"{experiment_tag}/00.loss/Train", scalar_value=training_loss[step], global_step=step)
                
                for epoch, recall in enumerate(dev_f1):
                    tb.add_scalar(f"{experiment_tag}/01.f1/development", scalar_value=dev_f1[epoch], global_step=epoch)
                
                for epoch, precision in enumerate(dev_precision):
                    tb.add_scalar(f"{experiment_tag}/02.precision/development", scalar_value=dev_precision[epoch], global_step=epoch)
                    
                for epoch, recall in enumerate(dev_recall):
                    tb.add_scalar(f"{experiment_tag}/03.recall/development", scalar_value=dev_recall[epoch], global_step=epoch)

                tb.close()
            

        # Scatter plot for dev and test
        if len(f1_eval) != 0 and args.scatter_plot: 
            eval_results[experiment_tag] = {"f1" : f1_eval.reshape(-1,2), 
                                            "recall": precision_eval.reshape(-1,2),
                                            "precision" : recall_eval.reshape(-1,2)}


            scatter_plt = plot_scatter(eval_results[experiment_tag], experiment_tag)

            if not os.path.exists(f"{output_dir}/scatter_plots"):
                os.mkdir(f"{output_dir}/scatter_plots")
                
            scatter_plt.savefig(f"{output_dir}/scatter_plots/dev_test_{experiment_tag}.png")
                    ####### COPY END #######


# ############### Robusted based on seeds ###############
# ==============================================================================


if __name__ == '__main__':
    run()