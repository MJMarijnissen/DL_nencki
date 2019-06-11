# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 12:05:29 2019

@author: Wladek
"""
import numpy as np
from NN_function import initialize_parameters, forward_propagation, compute_cost, backward_propagation, \
    update_parameters


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    np.random.seed(3)
    n_x = X.shape[0]
    n_y = Y.shape[0]
    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    parameters = initialize_parameters(n_x, n_h, n_y)
    for i in range(0,num_iterations):
        for b in range(0, X.shape[1],32):
            x_batch = X[:,b:b+32]
            y_batch = Y[:, b:b+32]
    # Loop (gradient descent)
        for i in range(0, num_iterations):
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
            A2, cache = forward_propagation(x_batch, parameters)
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
            cost = compute_cost(A2, y_batch, parameters)
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
            grads = backward_propagation(parameters, cache, x_batch, y_batch)
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
            parameters = update_parameters(parameters, grads)
        # Print the cost every 1000 iterations
            if print_cost and i % 1000 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
    return parameters
