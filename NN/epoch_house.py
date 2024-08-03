import pandas as pd
import numpy as np

# Function to return sigmoid of a number
def calculate_value_using_sigmoid(z):
    ans =  1 / (1 + np.exp(-z))
    return ans

# Function to normalise the dataset
def normalize(features):
    x_min = np.min(features)
    x_max = np.max(features)
    x_normal = (features-x_min)/(x_max-x_min)
    return x_normal

if __name__ == "__main__":

    # Reading data set
    data_frame = pd.read_csv('house_votes_84.csv')
    features = data_frame.iloc[:,:-1].values
    features = normalize(features)
    class_labels = data_frame['target'].values
    k_fold = 10
    fold_indices = []

    ## Hyper Parameters
    alpha = 0.1
    lamda = 0.2
    num_categories = 2

    # compute the class distribution of the target variable
    unique_classes, class_counts = np.unique(class_labels, return_counts=True)

    # Dividing the dataset based on its class value
    for unique_class in unique_classes:
        unique_class_indices = np.where(class_labels == unique_class)[0]
        np.random.shuffle(unique_class_indices)
        unique_class_folds = np.array_split(unique_class_indices, k_fold)
        fold_indices.extend(unique_class_folds)

    final_folds_list = []
    # Appending different class rows to get stratified cross validation
    for fold_index in range(k_fold):
        train_index = list(fold_indices[fold_index]) + list(fold_indices[(fold_index+k_fold)])
        final_folds_list.append(train_index)

    # Shuffling the indices of the folds with stratified cross validation
    for idx in final_folds_list:
        np.random.shuffle(idx)

    # List to store all the k-fold features
    k_fold_features_list = [] 
    # List to store all the k-fold labels
    k_fold_labels_list = []  
    for j in final_folds_list:
        temp_features=[]
        temp_labels=[]
        for l in j:
            temp_features.append(list(features[l]))
            temp_labels.append(class_labels[l])
        k_fold+=1
        k_fold_features_list.append(temp_features)
        k_fold_labels_list.append(temp_labels)

    # Define architecture of the neural network
    n_input = len(k_fold_features_list[0][0])  # number of features in input layer
    n_hidden = [4]  # list of number of neurons in each hidden layer
    n_output = 2  # number of neurons in output layer
    architecture = [n_input] + n_hidden + [n_output]  # list of layer sizes
    print("k_fold_features_list",k_fold_features_list)
    print('k_fold_labels_list ', k_fold_labels_list)
    print("Calculating average of all the testing folds...")

    testing_accuracy_mean_list = []
    testing_f1score_mean_list = []

    for no_of_times in range(10): # Runs 10 times where 1st fold is test set for first iteration and so on..
    
        gradient_theta_list_with_bias_term = []
        weighted_theta_list_with_bias_term = []

        for c in range(1, len(architecture)):
            rand_arr = np.random.normal(0, 1, size=(architecture[c], architecture[c-1]+1))
            # Assinging random weights from -1 to +1 for the first fold
            weighted_theta_list_with_bias_term.append(np.clip(rand_arr, -1, 1).tolist())
            # Assinging accumulated gradient array to Zeroes with same weighted array size
            gradient_theta_list_with_bias_term.append(np.clip(rand_arr, 0, 0).tolist())

        # Second stopping criterion mentioned in the assignment
        for sam in range(1000):
            # Counter to count number of elements in training set
            counter = 0
            weighted_theta_list_length = len(weighted_theta_list_with_bias_term)
            for i in range(len(k_fold_features_list)): # Runs 10 times as number of folds are 10
                if(i!=no_of_times): # Skips one time for test fold
                    for j in range(len(k_fold_features_list[i])): # Runs number of instances times in a particular fold
                        counter+=1
                        activation_result_list = [] # Stores activations of all neurons of all layers without bias term ( 1 ) in forward manner
                        delta_values_list = [] # Stores deltas of all neurons of all layer in backward manner
                        input = np.array(k_fold_features_list[i][j]).reshape(-1, 1)
                        input = np.insert(input, 0, 1, axis=0)
                        single_list = [item for sublist in input for item in sublist]
                        # Appending features in the activation list
                        activation_result_list.append(single_list[1:])

                        # Calculating activations of hidden and output layers
                        for k in range(len(weighted_theta_list_with_bias_term)):
                            result = np.dot(weighted_theta_list_with_bias_term[k],input)
                            for idx in result:
                                idx[0]=calculate_value_using_sigmoid(idx[0])
                            single_list = [item for sublist in result for item in sublist]
                            # Appending activation of the hidden and output neurons in the activation list
                            activation_result_list.append(single_list)
                            input = result
                            input = np.insert(input, 0, 1, axis=0)

                        # Using one hot encoding
                        one_hot_encoder_vector = np.zeros(num_categories)
                        one_hot_encoder_vector[k_fold_labels_list[i][j]] = 1
                        one_hot_encoder_vector = np.array(one_hot_encoder_vector).reshape(-1, 1)
                        temp_list = []
                        activation_list_length = len(activation_result_list)
                        activ = activation_result_list[activation_list_length-1]
                        activ = np.reshape(activ, (len(activ), 1))
                        answer  = (np.multiply(np.log(activ),-one_hot_encoder_vector))-(np.multiply(np.log(1-activ),1-one_hot_encoder_vector))
                        for idx in range(num_categories):
                            temp_list.append((activation_result_list[activation_list_length-1][idx])-one_hot_encoder_vector[idx][0])
                        # Appending delta values of output neurons
                        delta_values_list.append(temp_list)
                        delta_previous_list = delta_values_list[0]
                        input = np.array(delta_previous_list).reshape(-1, 1)
            
                        # Calculating delta values as part of backward propagation
                        for ki in range(weighted_theta_list_length-1, 0, -1):
                            result = np.dot(np.array(weighted_theta_list_with_bias_term[ki]).T,input)
                            result = result[1:]
                            activation_temp = np.array(activation_result_list[ki]).reshape(-1,1)
                            result = np.multiply(result, activation_temp)
                            result = np.multiply(result, 1-activation_temp)
                            result = [x[0] for x in result]
                            # Appending delta values of hidden layers in reverse manner
                            delta_values_list.append(result)
                            input = np.array(delta_values_list[len(delta_values_list)-1]).reshape(-1, 1)
                
                        # Calculating Gradients
                        delta_values_list_length = len(delta_values_list)-1
                        for ki in range(weighted_theta_list_length):
                            activation_input = np.reshape(activation_result_list[ki], (len(activation_result_list[ki]), 1))
                            activation_input = activation_input.tolist()
                            activation_input.insert(0, [1])
                            delta_input = delta_values_list[delta_values_list_length]
                            temp_gradient = (np.multiply(activation_input, delta_input))
                            temp_gradient = [[row[i] for row in temp_gradient] for i in range(len(temp_gradient[0]))]
                            gradient_theta_list_with_bias_term[ki] = np.add(gradient_theta_list_with_bias_term[ki], temp_gradient).tolist()
                            delta_values_list_length-=1

            # Calculating average of the accumulated derivates
            for idx in range(weighted_theta_list_length):
                P = (np.array(weighted_theta_list_with_bias_term[idx]) * lamda)
                P[:,0] = 0
                gradient_theta_list_with_bias_term[idx] = (gradient_theta_list_with_bias_term[idx]+P)/counter

            # Updating weights using accumulated gradients
            for idx in range(weighted_theta_list_length):
                weighted_theta_list_with_bias_term[idx] = weighted_theta_list_with_bias_term[idx] - (alpha * gradient_theta_list_with_bias_term[idx])

        # Length of the test fold
        fold_length = len(k_fold_features_list[no_of_times])
        TP = 0
        TN = 0
        no_of_correct_predictions = 0
        for ele in range(fold_length):
            one_hot_encoder_vector_test = np.zeros(num_categories)
            one_hot_encoder_vector_test[k_fold_labels_list[no_of_times][ele]] = 1
            one_hot_encoder_vector_test = np.array(one_hot_encoder_vector_test).reshape(-1, 1)
            activation_result_list_test = []
            input = np.array(k_fold_features_list[no_of_times][ele]).reshape(-1, 1)
            input = np.insert(input, 0, 1, axis=0)
            single_list = [item for sublist in input for item in sublist]
            activation_result_list.append(single_list[1:])

            # Calculating activations of hidden and output layers
            for k in range(weighted_theta_list_length):
                result = np.dot(weighted_theta_list_with_bias_term[k],input)
                for idx in result:
                    idx[0]=calculate_value_using_sigmoid(idx[0])
                single_list = [item for sublist in result for item in sublist]
                activation_result_list_test.append(single_list)
                input = result
                input = np.insert(input, 0, 1, axis=0)
            activation_result_list_test = activation_result_list_test[-1]
            max_index = np.argmax(activation_result_list_test)
            corresponding_element = one_hot_encoder_vector_test[max_index][0]
            if corresponding_element == 1:
                if max_index == 1:
                    TP += 1
                else:
                    TN += 1
                no_of_correct_predictions += 1

        total_positives = k_fold_labels_list[no_of_times].count(1)
        total_negatives = k_fold_labels_list[no_of_times].count(0)
        FN = total_positives - TP
        FP = total_negatives - TN
        precision = 0
        recall = 0
        f1score = 0
        if TP!=0:
            precision = TP/(TP+FP)
        if TP!=0:
            recall = TP/(TP+FN)
            f1score = (2*precision*recall)/(precision+recall)
            testing_f1score_mean_list.append(f1score)
        testing_accuracy_mean_list.append((no_of_correct_predictions*100)/fold_length) 

    print("average of testing_accuracy_mean_list",np.average(testing_accuracy_mean_list))
    print("average of testing_f1score_mean_list",np.average(testing_f1score_mean_list))
