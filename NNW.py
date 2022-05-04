import numpy as np
import random
import argparse

def __init__(self):
    np.random.seed(1)
def sigmoid(x):
    return 1/(1 + np.exp(-x))
def sigmoid_deriv(x):
    return sigmoid(x)*(1-sigmoid(x))

def init_weights_biases(num_input_nodes, num_hidden_nodes, num_output_nodes):
    weight_hidden = np.random.randn(num_hidden_nodes,num_input_nodes)
    weight_output = np.random.randn(num_output_nodes,num_hidden_nodes)
    hidden_biases = np.zeros((num_hidden_nodes,1))
    output_biases = np.zeros((num_output_nodes,1))
    parameter_dictionary = {"weight_hidden": weight_hidden,
                           "weight_output":weight_output,
                           "hidden_biases": hidden_biases,
                           "output_biases": output_biases}
    return parameter_dictionary

def read_file_to_array(file_name):
    with open(file_name) as input_file:
        read = input_file.readlines()
        feature_list_1 = []
        feature_list_2 = []
        header_list = []
        for i in read[0].split():
            header_list.append([i])
        label_list = []
        for line in read[1:]:
            line = line.replace("\n", "")
            for word in range(len(line)):
                if line[word].isnumeric() and word == 0:
                    feature_list_1.append(float(line[word]))
                if line[word].isnumeric() and word == 2:
                    feature_list_2.append(float(line[word]))
                elif line[word].isnumeric() and word == 4:
                    label_list.append(float(line[word]))
        label_list = [label_list]
        feature_array = np.array([feature_list_1, feature_list_2])
        label_array = np.array(label_list)
        header_array = np.array(header_list)
    return (feature_array, label_array, header_array)

def forward_propagate(feature_array, parameter_dictionary):
    output_vals = {}

    array = parameter_dictionary["weight_hidden"]
    hidden_b = parameter_dictionary["hidden_biases"]
    hidden_layer_values = np.dot(array,feature_array)+hidden_b
    activation  = sigmoid(hidden_layer_values)
    output_vals["hidden_layer_outputs"]=activation

    array_2 = parameter_dictionary["weight_output"]
    hidden_o = parameter_dictionary["output_biases"]
    hidden_layer_outputs = np.dot(array_2,feature_array)+hidden_o
    activation_2  = sigmoid(hidden_layer_outputs)
    output_vals["output_layer_outputs"]=activation_2

    return output_vals

def find_loss(output_layer_outputs, labels):
    num_examples = labels.shape[1]
    loss = (-1 / num_examples) * np.sum(np.multiply(labels, np.log(output_layer_outputs)) + np.multiply(1-labels, np.log(1-output_layer_outputs)))
    return loss

def backprop(feature_array, labels, output_vals, weights_biases_dict, verbose=False):
    if verbose:
        print()
    # We get the number of examples by looking at how many total
    # labels there are. (Each example has a label.)
    num_examples = labels.shape[1]
    # These are the outputs that were calculated by each
    # of our two layers of nodes that calculate outputs.
    hidden_layer_outputs = output_vals["hidden_layer_outputs"]
    output_layer_outputs = output_vals["output_layer_outputs"]
    # These are the weights of the arrows coming into our output
    # node from each of the hidden nodes. We need these to know
    # how much blame to place on each hidden node.
    output_weights = weights_biases_dict["weight_output"]
    # This is how wrong we were on each of our examples, and in
    # what direction. If we have four training examples, there
    # will be four of these.
    # This calculation works because we are using binary cross-entropy,
    # which produces a fairly simply calculation here.
    raw_error = output_layer_outputs - labels
    if verbose:
        print("raw_error", raw_error)
    # This is where we calculate our gradient for each of the
    # weights on arrows coming into our output.
    output_weights_gradient = np.dot(raw_error, hidden_layer_outputs.T)/num_examples
    if verbose:
        print("output_weights_gradient", output_weights_gradient)
    # This is our gradient on the bias. It is simply the
    # mean of our errors.
    output_bias_gradient = np.sum(raw_error, axis=1, keepdims=True)/num_examples

    if verbose:
        print("output_bias_gradient", output_bias_gradient)
        # We now calculate the amount of error to propagate back to our hidden nodes.
        # First, we find the dot product of our output weights and the error
        # on each of four training examples. This allows us to figure out how much,
        # for each of our training examples, each hidden node contributed to our
        # getting things wrong.
    blame_array = np.dot(output_weights.T, raw_error)
    if verbose:
        print("blame_array", blame_array)
        # hidden_layer_outputs is the actual values output by our hidden layer for
        # each of the four training examples. We square each of these values.
    hidden_outputs_squared = np.power(hidden_layer_outputs, 2)
    if verbose:
        print("hidden_layer_outputs", hidden_layer_outputs)
        print("hidden_outputs_squared", hidden_outputs_squared)
        # We now multiply our blame array by 1 minus the squares of the hidden layer's
        # outputs.
    propagated_error = np.multiply(blame_array, 1-hidden_outputs_squared)
    if verbose:
        print("propagated_error", propagated_error)
        # Finally, we compute the magnitude and direction in which we
        # should adjust our weights and biases for the hidden node.
    hidden_weights_gradient = np.dot(propagated_error, feature_array.T)/num_examples
    hidden_bias_gradient = np.sum(propagated_error, axis=1,keepdims=True)/num_examples
    if verbose:
        print("hidden_weights_gradient", hidden_weights_gradient)
        print("hidden_bias_gradient", hidden_bias_gradient)
        # A dictionary that stores all of the gradients
        # These are values that track which direction and by
        # how much each of our weights and biases should move
    gradients = {"hidden_weights_gradient": hidden_weights_gradient,
                "hidden_bias_gradient": hidden_bias_gradient,
                "output_weights_gradient": output_weights_gradient,
                "output_bias_gradient": output_bias_gradient}
    return gradients

def update_weights_biases(parameter_dictionary, gradients, learning_rate):

    hidden_weights = parameter_dictionary["weight_hidden"]
    output_weights = parameter_dictionary["weight_output"]
    hidden_bias = parameter_dictionary["hidden_biases"]
    output_bias = parameter_dictionary["output_biases"]

    hidden_weights_gradient = gradients['hidden_weights_gradient']
    hidden_bias_gradient = gradients['hidden_bias_gradient']
    output_weights_gradient = gradients['output_weights_gradient']
    output_bias_gradient = gradients['output_bias_gradient']
    updated_parameters = { "weight_hidden" : hidden_weights - learning_rate*hidden_weights_gradient,
                           "hidden_biases" : hidden_bias - learning_rate*hidden_bias_gradient,
                           "weight_output" : output_weights - learning_rate*output_weights_gradient,
                           "output_biases" : output_bias - learning_rate*output_bias_gradient}
    return updated_parameters

def model_file(file_name, num_inputs, num_hiddens, num_outputs, epochs, learning_rate):
    feature_array, label_array, header_array = read_file_to_array(file_name)
    parameter_dictionary = init_weights_biases(num_inputs,num_hiddens,num_outputs)
    epoch = 0
    while epoch <= epochs:
        forward_propgate = forward_propagate(feature_array,parameter_dictionary)
        vals_ForLoss = forward_propgate["output_layer_outputs"]
        loss = find_loss(vals_ForLoss,label_array)
        if epoch%100==0:
            print("LOSS",loss)
        gradients = backprop(feature_array, label_array, forward_propgate, parameter_dictionary,epoch%100==0)
        parameter_dictionary = update_weights_biases(parameter_dictionary, gradients, learning_rate)
        epoch+=1
    return parameter_dictionary

# print(model_file('xor.txt', 2, 2, 1, 1000, 0.1))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Neural Network')
    parser.add_argument('-i', '--inputFile', type = str, required = True)
    parser.add_argument('-n', '--num_inputs', type = int, required = True)
    parser.add_argument('-p', '--num_hiddens', type = int, required = False)
    parser.add_argument('-o', '--num_outputs', type = int, required = True)
    parser.add_argument('-e', '--epochs', type = int, required = True)
    parser.add_argument('-l', '--learning_rate', type = float, required = True)
    args = parser.parse_args()

    parameter_dictionary = model_file(args.inputFile, args.num_inputs, args.num_hiddens, args.num_outputs, args.epochs, args.learning_rate)
    print(parameter_dictionary)
