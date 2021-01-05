import numpy as np
import os

import json
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
import argparse

# ================================================================================================


flatten = lambda t: [item for sublist in t for item in sublist]


def tsne_2d(x, y, title, labels):
    """
    Returns a matplotlib figure containing the 2D T-SNE plot.
    
    Args:
        x, y, z: arrays
        title: string with name of the plot
        labels: list of strings with label names: [x, y, z]
    """
    plt.rcParams.update({'font.size': 60, 'legend.fontsize': 20})
    plt.rc('font', size=50)
    plt.rc('axes', titlesize=50)

    figure, ax = plt.subplots(figsize=(14, 12))
    ax.scatter(x, y)
    ax.set_title(title) 
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    plt.grid()
    
    plt.tight_layout()

    return figure    


def tsne_3d(x, y, z, title, labels):
    """
    Returns a matplotlib figure containing the 3D T-SNE plot.
    
    Args:
        x, y, z: arrays
        title: string with name of the plot
        labels: list of strings with label names: [x, y, z]
    """
    plt.rcParams.update({'font.size': 30, 'legend.fontsize': 20})
    plt.rc('font', size=25)
    plt.rc('axes', titlesize=25)

    figure = plt.figure(figsize=(18,12))
    ax = figure.add_subplot(projection='3d')
    ax.scatter(one, two, thr)
    ax.set_title(title) 
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    plt.tight_layout()

    return figure


# ================================================================================================

# Define path to data source: 

parser = argparse.ArgumentParser()

parser.add_argument("--data-dir", default="data/outputs/seed_experiment_500")
parser.add_argument("--output-dir", default="luke_meta_analysis")
parser.add_argument('--save-tsne', dest='tsne_save', action='store_true')
parser.add_argument('--no-save-tsne', dest='tsne_save', action='store_false')
parser.set_defaults(tsne_save=True)

args = parser.parse_args()

data_dir = args.data_dir
output_dir = args.output_dir
tsne_save = args.tsne_save


#
# data_dir = "../data/outputs/seed_experiment_500"
# output_dir = "."
# tsne_save = True
#

def run():
    t_sne_dir = os.path.join(output_dir, "t_sne_plots")

    logits = []
    for root, dirs, files in os.walk(data_dir):
        
        for dir_ in dirs: 

            result_json = os.path.join(root, dir_, "results.json")
            base_root = os.path.basename(root)

            with open(result_json) as json_file:
                data = json.load(json_file)

            evaluations = data["evaluation_predict_label"]
            dev_test = {"dev": {}, "test": {}}
            eval_sets = ['dev', 'test']

            logits.append(flatten(evaluations["dev"]["predict_logits"]))


    matrix = np.array(flatten(logits)).reshape(-1, len(logits))
    data_matrix = TSNE(n_components=3).fit_transform(matrix)

    # Define compoments: 
    one = data_matrix[:,0]
    two = data_matrix[:,1]
    thr = data_matrix[:,2]

    title = "T-SNE - Seed Experiment"

    # 2D: 
    one_two = tsne_2d(one, two, title=title, labels=["1st T-SNE component", "2nd T-SNE component"])
    one_thr = tsne_2d(one, thr, title=title, labels=["1st T-SNE component", "3rd T-SNE component"])
    two_thr = tsne_2d(two, thr, title=title, labels=["2nd T-SNE component", "3rd T-SNE component"])

    # 3D: 
    one_two_thr = tsne_3d(one, two, thr, title, ["1st T-SNE component", "2nd T-SNE component", "3rd T-SNE component"])


    if not os.path.exists(t_sne_dir):
        os.mkdir(os.path.join(t_sne_dir)

    if tsne_save: 
        one_two.savefig(f"{t_sne_dir}/one_two")
        one_thr.savefig(f"{t_sne_dir}/one_thr")
        two_thr.savefig(f"{t_sne_dir}/two_thr")

        one_two_thr.savefig(f"{t_sne_dir}/one_two_thr")

if __name__ == "__main__":
    run()
