import numpy as np

from mutensor.tensor import Tensor

from typing import Tuple

def confusion_matrix(y_true:Tensor,y_pred:Tensor) -> Tuple[float,float,float,float] :
    """
    Returns the confusion matrix of predictions and truth values in a tuple .
    True Positives , False Positives , True Negatives, False Negatives

    """

    true_positives = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    false_positives = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    true_negatives = np.sum(np.logical_and(y_pred==0,y_true == 0))
    false_negatives = np.sum(np.logical_and(y_pred==0,y_true == 1))


    return (true_positives,false_positives,true_negatives,false_negatives)



def precision(y_true:Tensor,y_pred:Tensor) -> float :
    """
        Precision measures the accuracy of a classifier using the following formula

        precision = tp / (tp + fp) .

        tp : True Positive i.e Prediction was True and Label was True
        fp : False Positive i.e Prediction was True and Label was false
        
    """
    
    (tp,fp,tn,fn) = confusion_matrix(y_true,y_pred)

    precision = tp / (tp + fp)

    return precision
