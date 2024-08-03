import pandas as pd
import numpy as np
import math

def calculate_value_using_sigmoid(z):
    ans =  1 / (1 + np.exp(-z))
    return ans

if __name__ == "__main__":

    lamda = 0.25
    # Define architecture of the neural network
    n_input = 2  # number of features in input layer
    n_hidden = [4, 3]  # list of number of neurons in each hidden layer
    n_output = 2  # number of neurons in output layer
    architecture = [n_input] + n_hidden + [n_output]  # list of layer sizes
    print("architecture",architecture)

    k_fold_features_list = [
                                [
                                    [0.32000,0.68000],
                                    [0.83000,0.02000]
                                ]
                            ]
    
    k_fold_labels_list = [
                            [
                                [0.75000,0.98000],
                                [0.75000,0.28000]
                            ]
                        ]


    # Define weight matrices and bias vectors for each layer
    weighted_theta_list_with_bias_term = [
                                        [[0.42,0.15,0.4],[0.72,0.1,0.54],[0.01,0.19,0.42],[0.3,0.35,0.68]],
                                        [[0.21,0.67,0.14,0.96,0.87],[0.87,0.42,0.2,0.32,0.89],[0.03,0.56,0.8,0.69,0.09]],
                                        [[0.04,0.87,0.42,0.53],[0.17,0.1,0.95,0.69]]
                                        ]

    print("weighted_theta_list_with_bias_term",weighted_theta_list_with_bias_term)
    print()

    for i in range(1):
        J = 0
        gradient_theta_list_with_bias_term = []

        # Assinging accumulated gradient array to Zeroes with same weighted array size
        for c in range(1, len(architecture)):
            rand_arr = np.random.normal(0, 1, size=(architecture[c], architecture[c-1]+1))
            gradient_theta_list_with_bias_term.append(np.clip(rand_arr, 0, 0).tolist())

        for j in range(2):
            print("For instance",j+1)
            activation_result_list = [] # Stores activations of all neurons of all layers without bias term ( 1 ) in forward manner
            delta_values_list = [] # Stores deltas of all neurons of all layer in backward manner
            input = np.array(k_fold_features_list[i][j]).reshape(-1, 1)
            input = np.insert(input, 0, 1, axis=0)
            single_list = [item for sublist in input for item in sublist]
            activation_result_list.append(single_list[1:])
            print("Activation without sigmoid",1,input)

            # Calculating activations of hidden and output layers
            for k in range(len(weighted_theta_list_with_bias_term)):
                result = np.dot(weighted_theta_list_with_bias_term[k],input)
                print("Activation without sigmoid",k+2,result)
                for idx in result:
                    idx[0]=calculate_value_using_sigmoid(idx[0])
                single_list = [item for sublist in result for item in sublist]
                activation_result_list.append(single_list)
                input = result
                input = np.insert(input, 0, 1, axis=0)

            print("Activation with sigmoid",j+1,activation_result_list)
            print()
            temp_list = []
            activation_list_length = len(activation_result_list)
            activ = activation_result_list[activation_list_length-1]
            y = k_fold_labels_list[i][j]
            answer = np.multiply(np.log(activ),-np.array(y))-(np.multiply(np.log(1-np.array(activ)),1-np.array(y)))
            print("cost J of instance",j+1,np.sum(answer))
            J += np.sum(answer)
            temp_list.append(np.array(activ)-np.array(y))
            delta_values_list.append(temp_list)
            weighted_theta_dict_length = len(weighted_theta_list_with_bias_term)
            delta_previous_list = delta_values_list[0]
            input = np.array(delta_previous_list).reshape(-1, 1)

            # Calculating delta values as part of backward propagation
            for ki in range(weighted_theta_dict_length-1, 0, -1):
                result = np.dot(np.array(weighted_theta_list_with_bias_term[ki]).T,input)
                result = result[1:]
                activation_temp = np.array(activation_result_list[ki]).reshape(-1,1)
                result = np.multiply(result, activation_temp)
                result = np.multiply(result, 1-activation_temp)
                result = [x[0] for x in result]
                delta_values_list.append(result)
                input = np.array(delta_values_list[len(delta_values_list)-1]).reshape(-1, 1)

            print("delta_values_list",delta_values_list)
            print()
            # Calculating Gradients
            delta_values_list_length = len(delta_values_list)-1
            for ki in range(weighted_theta_dict_length):
                activation_input = np.reshape(activation_result_list[ki], (len(activation_result_list[ki]), 1))
                activation_input = activation_input.tolist()
                activation_input.insert(0, [1])
                delta_input = delta_values_list[delta_values_list_length]
                temp_gradient = (np.multiply(activation_input, delta_input))
                temp_gradient = [[row[i] for row in temp_gradient] for i in range(len(temp_gradient[0]))]
                print("Gradients of Theta",ki+1,temp_gradient)
                print()
                gradient_theta_list_with_bias_term[ki] = np.add(gradient_theta_list_with_bias_term[ki], temp_gradient).tolist()
                delta_values_list_length-=1

        # Calculating the term S ( Squares of all weights except bias weights)
        S = 0
        J /= len(k_fold_features_list[i])
        for idx in range(weighted_theta_dict_length):
            S += sum([sum([num ** 2 for num in row[1:]]) for row in weighted_theta_list_with_bias_term[idx]])

        S = S * (lamda/(2*len(k_fold_features_list[i])))
        J += S
        print("Final (regularized) cost, J = ",J)
        # Calculating average of the accumulated derivates
        for idx in range(weighted_theta_dict_length):
            P = (np.array(weighted_theta_list_with_bias_term[idx]) * lamda)
            P[:,0] = 0
            gradient_theta_list_with_bias_term[idx] = (gradient_theta_list_with_bias_term[idx]+P)/len(k_fold_features_list[i])

        print("Final regularized gradients",gradient_theta_list_with_bias_term)