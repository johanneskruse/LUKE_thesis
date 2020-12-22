import numpy as np

from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
plt.rc('font', size=18)
plt.rc('axes', titlesize=20)



def plot_scatter(eval_dict, title, labels=["Development", "Test"]):
    """
    Returns a matplotlib figure containing the plotted scatter plot.
    
    Args:
        eval_dict: dictionary with evaluation matrices
        title: string with name of the CM
        labels: list of string x and y axis name 
    """

    figure = plt.figure(figsize=(10,7))
                
    max_min = []
    for tag in eval_dict:
        dev = eval_dict[tag][:,0]
        test = eval_dict[tag][:,1]

        plt.scatter(dev, test, label=f"{tag}")

        max_min.extend([min(dev), min(test), max(dev), max(test)])

    dummy = np.linspace(min(max_min)-0.01,  max(max_min)+0.01, 2)
    plt.plot(dummy, dummy, linestyle = '--', label = 'Identity line', color="black")
                
    plt.title(title, fontsize=24)
    plt.grid()
    plt.legend()
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.tight_layout()
    return figure


def plot_calibration_curve(y_true, y_pred_prob, model_name, title, n_bins=5, normalize=False):
    """
    Returns a matplotlib figure containing the plotted calibration plot.
    
    Args:
        y_true (array, shape = [n, n]): binary matrix with true labels
        y_pred_prob (array, shape = [n, n]): Probability matrix with predicted labels
        model_name: name of model(s) to be plotted
        normalize: Whether y_prob needs to be normalized into the [0, 1] interval
        n_bins: Number of bins to discretize the [0, 1] interval. 
    """

    figure = plt.figure(figsize=(10,7))
    
    # Ideal calibration line: 
    plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Ideally Calibrated', color="black") 

    if type(model_name) == str: 
        x, y = calibration_curve(y_true, y_pred_prob, n_bins=n_bins, normalize=normalize)
        plt.plot(x, y, marker = '.', label = model_name)
    else: 
        for i in range(len(model_name)):
            x, y = calibration_curve(y_true[i], y_pred_prob[i], n_bins=n_bins, normalize=normalize)
            plt.plot(x, y, marker = '.', label = model_name[i]) 

    plt.plot([0, 1], [0, 1], linestyle = '--', color="black") 

    plt.title(title, fontsize=24)
    plt.grid()
    plt.legend() 
    plt.xlabel('Average Predicted Probability in each bin') 
    plt.ylabel('Ratio of positives') 
    plt.tight_layout()
    return figure


def logit2prob(logits):
    prob = np.array([])
    for logit in logits: 
        prob = np.append(prob, (np.exp(logit) / (np.exp(logit) + 1)))
    return prob