import numpy as np
import random 

def __init__(self):
    np.random.seed(1)
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_deriv(x):
    return sigmoid(x)*(1-sigmoid(x))
    
