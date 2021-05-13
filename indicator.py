import numpy as np

def DICE(trueLabel, result):
    intersection=np.sum(np.minimum(np.equal(trueLabel,result),trueLabel))
    union = np.count_nonzero(trueLabel)+np.count_nonzero(result)
    dice = 2 * intersection / (union + 10**(-9))
   
    return dice

def recall(true, pred):
    eps = 10**-9
    tp_fn = true.sum()
    tp    = ((true == pred) * true).sum()

    score = tp / (tp_fn + eps)

    return score
