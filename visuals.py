###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score


def distribution(data, colnames_list, transformed = False):
    """
    Visualization code for displaying skewed distributions of features
    """
    
    if transformed:
        rows, cols, width, height = 4, 3, 15, 15
    else:
        rows, cols, width, height = 15, 3, 15, 50
    
    # Create figure
    fig = pl.figure(figsize = (width,height));

    # Skewed feature plotting
    for i, feature in enumerate(colnames_list):
        ax = fig.add_subplot(rows, cols, i+1)
        visual_data = data[feature]
        if not transformed:
            upper = np.percentile(data[feature], 98, axis=0)
            visual_data = visual_data[visual_data<=upper]
        ax.hist(visual_data, bins = 20,color='C1',edgecolor='white')
        ax.set_title("'%s' Feature Distribution"%(feature), fontsize = 14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        ax.set_ylim((0, 8000))
        ax.set_yticks([0, 2000, 4000, 6000, 8000])
        ax.set_yticklabels([0, 2000, 4000, 6000, ">8000"])

    # Plot aesthetics
    if transformed:
        fig.suptitle("Log-transformed Distributions of Continuous Data Features", \
            fontsize = 16, y = 1.03)
    else:
        fig.suptitle("Distributions of Continuous Data Features", \
            fontsize = 16, y = 1.03)

    fig.tight_layout()
    if not transformed:
        fig.savefig('before_transformed.jpg')
    else:
        fig.savefig('after_transformed.jpg')
    fig.show()


def evaluate(results, accuracy, f1):
    """
    Visualization code to display results of various learners.
    
    inputs:
      - learners: a list of supervised learners
      - stats: a list of dictionaries of the statistic results from 'train_predict()'
      - accuracy: The score for the naive predictor
      - f1: The score for the naive predictor
    """
  
    # Create figure
    fig, ax = pl.subplots(2, 3, figsize = (11,7))

    # Constants
    #bar_width = 0.3
    #colors = ['#A00000','#00A0A0','#00A000']
    
    # Super loop to plot four panels of data
    #for k, learner in enumerate(results.keys()):
    #    for j, metric in enumerate(['train_time', 'train_acc', 'train_auc', 'pred_time', 'test_acc', 'test_auc']):
    #        for i in np.arange(3):
                
                # Creative plot code
    #            ax[j/3, j%3].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])
    #            ax[j/3, j%3].set_xticks([0.45, 1.45, 2.45])
    #            ax[j/3, j%3].set_xticklabels(["1%", "10%", "100%"])
    #            ax[j/3, j%3].set_xlabel("Training Set Size")
    #            ax[j/3, j%3].set_xlim((-0.1, 3.0))
    
    bar_width =0.2
    colors = ['#5c0d2d','#fcdee3','#a0a0a0','#121212']
    
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'train_acc', 'train_fscore',
                                    'pred_time', 'valid_acc', 'valid_fscore']):
            for i in np.arange(3):
                
                # Creative plot code
                ax[j/3, j%3].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])
                ax[j/3, j%3].set_xticks([0.45, 1.45, 2.45])
                ax[j/3, j%3].set_xticklabels(["5%", "20%", "100%"])
                ax[j/3, j%3].set_xlabel("Training Set Size")
                ax[j/3, j%3].set_xlim((-0.1, 3.0))
    
    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F1_Score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F1_Score")
    
    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F1_Score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Validation Set")
    ax[1, 2].set_title("F1_Score on Validation Set")
    
    # Add horizontal lines for naive predictors
    ax[0, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 1].axhline(y = accuracy, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[0, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    ax[1, 2].axhline(y = f1, xmin = -0.1, xmax = 3.0, linewidth = 1, color = 'k', linestyle = 'dashed')
    
    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color = colors[i], label = learner))
    pl.legend(handles = patches, bbox_to_anchor = (-.80, 2.70), \
               loc = 'upper center', borderaxespad = 0., ncol = 4, fontsize = 'x-large')
    
    # Aesthetics
    pl.suptitle("Performance Metrics for Four Supervised Learning Models", fontsize = 16, y = 1.10)
    pl.tight_layout(rect=[0, 0.03, 1, 0.95])
    pl.savefig('model.jpg')
    pl.show()
    
    



def feature_plot(importances, X_train, y_train):
    
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Creat the plot
    fig = pl.figure(figsize = (9,5))
    pl.title("Normalized Weights for First Five Most Predictive Features", fontsize = 16)
    pl.bar(np.arange(5), values, width = 0.6, align="center", color = '#00A000', \
          label = "Feature Weight")
    pl.bar(np.arange(5) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', \
          label = "Cumulative Feature Weight")
    pl.xticks(np.arange(5), columns)
    pl.xlim((-0.5, 4.5))
    pl.ylabel("Weight", fontsize = 12)
    pl.xlabel("Feature", fontsize = 12)
    
    pl.legend(loc = 'upper center')
    pl.tight_layout()
    pl.show()  
