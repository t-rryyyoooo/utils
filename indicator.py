import numpy as np
from sklearn.metrics import recall_score, precision_score 

def DICE(trueLabel, result):
    intersection=np.sum(np.minimum(np.equal(trueLabel,result),trueLabel))
    union = np.count_nonzero(trueLabel)+np.count_nonzero(result)
    dice = 2 * intersection / (union + 10**(-9))
   
    return dice

def recall(true_array: np.ndarray, pred_array: np.ndarray):
    score = recall_score(true_array, pred_array, average="macro")

    return score

def precision(true_array: np.ndarray, pred_array: np.ndarray):
    score = precision_score(true_array, pred_array, average=None)

    return score
