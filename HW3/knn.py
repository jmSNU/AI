import numpy as np
import matplotlib.pyplot as plt


def load_knn_data():
    test_inputs = np.genfromtxt('knn-dataset/test_inputs.csv', delimiter=','),
    test_labels = np.genfromtxt('knn-dataset/test_labels.csv', delimiter=','),
    train_inputs = np.genfromtxt('knn-dataset/train_inputs.csv', delimiter=','),
    train_labels = np.genfromtxt('knn-dataset/train_labels.csv', delimiter=','),
    return train_inputs, train_labels, test_inputs, test_labels


'''
This function implements the KNN classifier to predict the label of a data point. 
Measure distances with the Euclidean norm (L2 norm).  
When there is a tie between two (or more) labels, break the tie by choosing any label.

Inputs:
    **x**: input data point for which we want to predict the label (numpy array of M features)
    **inputs**: matrix of data points in which neighbors will be found (numpy array of N data points x M features)
    **labels**: vector of labels associated with the data points  (numpy array of N labels)
    **k_neighbors**: # of nearest neighbors that will be used
Outputs:
    **predicted_label**: predicted label (integer)
'''   
def predict_knn(x, inputs, labels, k_neighbors):
    predicted_label = 0
    ########
    # TO DO:
    try: # Due to the typo in the load_knn_data(), I couldn't help but add this exception
        N = inputs.shape[0]
    except:
        N = inputs[0].shape[0]
        inputs = inputs[0]
        labels = labels[0]
    dist_arr = np.zeros((N,))
    for i in range(0,N):
        dist_arr[i] = np.sum((x - inputs[i])**2) # for the performance I used square of the L2 Norm
        
    sorted_idx = sorted(list(range(len(dist_arr))),key = lambda k:dist_arr[k],reverse=False) # sorting the index w.r.t the distances
   
    neighbors = sorted_idx[:k_neighbors]
    label_of_neighbor = [labels[k] for k in neighbors] # label of the close features

    neighbor_labels,cnt = np.unique(label_of_neighbor,return_counts=True) # cnt has the number of each label
    predicted_label = neighbor_labels[np.argmax(cnt)] # get the maximum of cnt. Due to the characteristic of np.argmax, if the tie happened, it will return the foremost label
    ########
    return predicted_label



'''
This function evaluates the accuracy of the KNN classifier on a dataset. 
The dataset to be evaluated consists of (inputs, labels). 
The dataset used to find nearest neighbors consists of (train_inputs, train_labels).

Inputs:
   **inputs**: matrix of input data points to be evaluated (numpy array of N data points x M features)
   **labels**: vector of target labels for the inputs (numpy array of N labels)
   **train_inputs**: matrix of input data points in which neighbors will be found (numpy array of N' data points x M features)
   **train_labels**: vector of labels for the training inputs (numpy array of N' labels)
   **k_neighbors**: # of nearest neighbors to be used (integer)
Outputs:
   **accuracy**: percentage of correctly labeled data points (float)
'''
def eval_knn(inputs, labels, train_inputs, train_labels, k_neighbors):
    accuracy = 0
    ########
    # TO DO:
    try:
        N = inputs.shape[0]
    except:
        N = inputs[0].shape[0]
        inputs = inputs[0]
        labels = labels[0]
        train_inputs = train_inputs[0]
        train_labels = train_labels[0]
    num_correct = 0
    for i in range(N):
        x = inputs[i] # target
        x_label = labels[i] # true label
        x_label_predicted = predict_knn(x,train_inputs,train_labels,k_neighbors) # predicted label
        if x_label_predicted == x_label:
            num_correct+=1
    accuracy = num_correct/N
    ########
    return accuracy


'''
This function performs k-fold cross validation to determine the best number of neighbors for KNN.
        
Inputs:
    **k_folds**: # of folds in cross-validation (integer)
    **hyperparameters**: list of hyperparameters where each hyperparameter is a different # of neighbors (list of integers)
    **inputs**: matrix of data points to be used when searching for neighbors (numpy array of N data points by M features)
    **labels**: vector of labels associated with the inputs (numpy array of N labels)
Outputs:
    **best_hyperparam**: best # of neighbors for KNN (integer)
    **best_accuracy**: accuracy achieved with best_hyperparam (float)
    **accuracies**: vector of accuracies for the corresponding hyperparameters (numpy array of floats)
'''
def cross_validation_knn(k_folds, hyperparameters, inputs, labels):
    best_hyperparam = 0
    best_accuracy = 0
    accuracies = np.zeros(len(hyperparameters))
    ########
    # TO DO:
    try:
        N = inputs.shape[0]
    except:
        N = inputs[0].shape[0]
        inputs = inputs[0]
        labels = labels[0]
    for j in range(len(hyperparameters)):
        tmp_acc = np.zeros(k_folds) 
        for i in range(k_folds):
            valid_sets = inputs[i*N//k_folds:(i+1) * N//k_folds] # validation set is a portion of train set
            valid_label_sets = labels[i*N//k_folds:(i+1) * N//k_folds] # the label of validation set
            input_sets = np.concatenate((inputs[:i*N//k_folds],inputs[(i+1)*N//k_folds:]),axis = 0) # the remaining data is regarded as train set
            input_label_sets = np.concatenate((labels[:i*N//k_folds],labels[(i+1)*N//k_folds:]),axis = 0) # the label of train set
            tmp_acc[i] = eval_knn(valid_sets,valid_label_sets,input_sets,input_label_sets,hyperparameters[j]) # k_folds number of accuracies must be calculated
        accuracies[j] = np.mean(tmp_acc) # in cross_validation method, the accuracy of each hyperparameter is obtained by the average of k_folds number of accuracies(From what I've looked for, one can use the other index of estimation and representative value. However, I just used very naive one)
    idx = np.argmax(accuracies)
    best_accuracy = accuracies[idx]
    best_hyperparam = hyperparameters[idx]
    ########
    return best_hyperparam, best_accuracy, accuracies


'''
This function plots the KNN accuracies for different # of neighbors (hyperparameters) based on cross validation

Inputs:
    **accuracies**: vector of accuracies for the corresponding hyperparameters (numpy array of floats)
    **hyperparams**: list of hyperparameters where each hyperparameter is a different # of neighbors (list of integers)
'''
def plot_knn_accuracies(accuracies, hyperparams):
    plt.plot(hyperparams, accuracies)
    plt.ylabel('accuracy')
    plt.xlabel('k neighbors')
    plt.show()
    
def main():
    # load data
    train_inputs, train_labels, test_inputs, test_labels = load_knn_data()    
    # number of neighbors to be evaluated by cross validation
    hyperparams = range(1,31)
    k_folds = 10

    # use k-fold cross validation to find the best # of neighbors for KNN
    best_k_neighbors, best_accuracy, accuracies = cross_validation_knn(k_folds, hyperparams, train_inputs, train_labels)

    # plot results
    plot_knn_accuracies(accuracies, hyperparams)
    print('best # of neighbors k: ' + str(best_k_neighbors))
    print('best cross validation accuracy: ' + str(best_accuracy))

    # evaluate with best # of neighbors
    accuracy = eval_knn(test_inputs, test_labels, train_inputs, train_labels, best_k_neighbors)
    print('test accuracy: '+ str(accuracy))

if __name__ == "__main__":
    main()