import numpy as np


class Cross_Entropy_Cost:


    #the actual cost function
    @staticmethod
    def fn(a, y):
        np.sum(np.nan_to_num(-y * np.log(a) - (1-y) * np.log(1-a)))

    """how to compute the output error for this cost function
    first param is included to support other cost functions with different delta methods"""

    @staticmethod
    def delta(_, a, y):
        return a-y