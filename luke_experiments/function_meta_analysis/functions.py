import numpy as np

from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 30, 'legend.fontsize': 20})
plt.rc('font', size=25)
plt.rc('axes', titlesize=25)


flatten = lambda t: [item for sublist in t for item in sublist]


def name_changes(dict_org, dict_names):
    for label_name in dict_org.keys():
        if label_name in dict_names.keys(): 
            dict_org[dict_names[label_name]] = dict_org.pop(label_name)
    
    for label_name in dict_org.keys():
        if label_name in dict_names.keys(): 
            dict_org[dict_names[label_name]] = dict_org.pop(label_name)
    
    return dict_org


def plot_f1(f1_scores, labels, title):    
    figure, ax = plt.subplots(figsize=(12,9))

    for i, experiment in enumerate(sorted(f1_scores)):
        ax.plot(f1_scores[experiment], marker=".")

    ax.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5), 
                edgecolor="white", title="Model", fontsize="medium")

    ax.set_xticks(range(0,len(f1_scores[experiment]))) 
    ax.set_xticklabels(range(1,len(f1_scores[experiment])+1))
    ax.set_title(f"F1-score training\n{title}", size="x-large") 
    
    ax.set_xlabel("Epoch", fontsize="large")
    ax.set_ylabel("F1-score", fontsize="large")

    plt.tick_params(axis='x', labelsize="large")
    plt.tick_params(axis='y', labelsize="large")
    plt.grid()
    figure.tight_layout(rect=[0, 0.03, 1, 0.95])
    #plt.tight_layout()

    return figure

def plot_scatter(eval_dict, labels, title, axes=["Development", "Test"]):
    """
    Returns a matplotlib figure containing the plotted scatter plot.
    
    Args:
        eval_dict: dictionary with evaluation matrices
        labels: experimental tag
        title: string with name of the CM
        axes: list of string x and y axis name 
    """
    # ===== Parameter settings =====
    markers = [".", ",", "o", "v", "^", "<", ">", "1", "2",
                "3", "4", "8", "s", "p", "P", "*", "h", "H", 
                "+", "x", "X", "D", "d"]
    color = ["black", "blue", "orange", "green"]
    marker_size = 200
    metrics = ['Identity line']
    metrics.extend(list(eval_dict))

    max_min = []

    # ===== Start Figure =====
    figure, ax = plt.subplots(figsize=(14,9))
    
    for z, tag in enumerate(eval_dict):
        dev = eval_dict[tag][:,0]
        test = eval_dict[tag][:,1]

        # ===== Add samples row by row =====
        for i in range(len(dev)): 
            ax.scatter(dev[i], test[i], marker=markers[i], color=color[z+1], s=marker_size)

        max_min.extend([min(dev), min(test), max(dev), max(test)])

    # ===== Legend =====
    # First legend: Models
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.6, box.height])

    legend1 = ax.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5), 
                        edgecolor="white", title="Model symbol", fontsize="medium")
    ax.add_artist(legend1)
    
    leg = ax.get_legend()
    for legendH in leg.legendHandles: 
        legendH.set_color('black')

    dummy = np.linspace(min(max_min)-0.01,  max(max_min)+0.01, 2)
    for z in range(len(color)):
        if color[z] == "black": 
            ax.plot(dummy, dummy, linestyle = '--', color=color[z])
        else:
            ax.plot(dummy, dummy, linestyle = 'solid', color=color[z])
    
    ax.plot(dummy, dummy, linestyle = "solid", color="white")
    ax.plot(dummy, dummy, linestyle = "--", color="black")

    # Legend: Model metric:
    legend2 = ax.legend(metrics, loc='upper left', edgecolor="blue", fontsize="medium")
    ax.add_artist(legend2)

    leg = ax.get_legend()
    for i, legendH in enumerate(leg.legendHandles): 
        legendH.set_color(color[i])
    
    # ===== Finishing touch =====
    # Add: Title, grid, axes-naming, and tighten plot
    ax.set_title(title, size="x-large") 
    ax.set_xlabel(axes[0], fontsize="large")
    ax.set_ylabel(axes[1], fontsize="large")
    plt.tick_params(axis='x', labelsize="large")
    plt.tick_params(axis='y', labelsize="large")
    plt.grid()
    plt.tight_layout()
    return figure


def plot_calibration_curve(y_true, y_pred_prob, model_name, title, n_bins=5, normalize=False):
    """
    Returns a matplotlib figure containing the plotted calibration plot.
    
    Args:
        y_true: array-like of shape (n_samples,) of true target
        y_pred_prob: array-like of shape (n_samples,) probabilities of the positive class.
        model_name: name of model(s) to be plotted
        normalize: Whether y_prob needs to be normalized into the [0, 1] interval
        n_bins: Number of bins to discretize the [0, 1] interval. 
    """

    figure = plt.figure(figsize=(14,9))
    
    # Ideal calibration line: 
    plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Ideally Calibrated', color="black") 

    if type(model_name) == str: 
        x, y = calibration_curve(y_true, y_pred_prob, n_bins=n_bins, normalize=normalize)
        plt.plot(x, y, marker = '.', label = model_name)
    else: 
        for i in range(len(model_name)):
            fraction_of_positives, mean_predicted_value = calibration_curve(y_true[i], y_pred_prob[i], n_bins=n_bins, normalize=normalize)
            plt.plot(mean_predicted_value, fraction_of_positives, marker = '.', label = model_name[i]) 

    plt.plot([0, 1], [0, 1], linestyle = '--', color="black") 

    plt.title(title, size="x-large")
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="medium")
    plt.tick_params(axis='x', labelsize="large")
    plt.tick_params(axis='y', labelsize="large")
    plt.xlabel('Mean predicted value', fontsize="large") 
    plt.ylabel('Fraction of positives', fontsize="large") 
    plt.tight_layout()
    return figure


def plot_calibration_curve_with_hist(y_true, y_pred_prob, model_name, title, n_bins=10, normalize=False):
    """
    Returns a matplotlib figure containing the plotted calibration plot.
    
    Args:
        y_true: array-like of shape (n_samples,) of true target
        y_pred_prob: array-like of shape (n_samples,) probabilities of the positive class.
        model_name: name of model(s) to be plotted
        normalize: Whether y_prob needs to be normalized into the [0, 1] interval
        n_bins: Number of bins to discretize the [0, 1] interval. 
    """

    figure = plt.figure(figsize=(14,10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    
    # Ideal calibration line: 
    ax1.plot([0, 1], [0, 1], linestyle = '--', label = 'Ideally Calibrated', color="black") 

    if type(model_name) == str: 
        x, y = calibration_curve(y_true, y_pred_prob, n_bins=n_bins, normalize=normalize)
        plt.plot(x, y, marker = '.', label = model_name)
    else: 
        for i in range(len(model_name)):
            fraction_of_positives, mean_predicted_value = calibration_curve(y_true[i], y_pred_prob[i], n_bins=n_bins, normalize=normalize)
            ax1.plot(mean_predicted_value, fraction_of_positives, marker = '.', label = model_name[i]) 
            ax2.hist(y_pred_prob[i], range=(0, 1), bins=n_bins, label=model_name, histtype="step", lw=2)

    ax1.plot([0, 1], [0, 1], linestyle = '--', color="black") 

    ax1.set_title(title, size="x-large")
    ax1.grid()
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="medium")
    ax1.tick_params(axis='x', labelsize="large")
    ax1.tick_params(axis='y', labelsize="large")
    ax1.set_ylabel('Fraction of positives', fontsize="large") 

    ax2.tick_params(axis='x', labelsize="large")
    ax2.tick_params(axis='y', labelsize="large")
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count", fontsize="large")

    plt.tight_layout()
    return figure


def logit2prob(logits):
    prob = np.array([])
    for logit in logits: 
        prob = np.append(prob, (np.exp(logit) / (np.exp(logit) + 1)))
    return prob

