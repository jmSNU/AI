import numpy as np


def load_data_small():
    """ Load small training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An N_train-x-M ndarray containing the training data (N_train examples, M features each)
        y_train: An N_train-x-1 ndarray contraining the labels
        X_val: An N_val-x-M ndarray containing the training data (N_val examples, M features each)
        y_val: An N_val-x-1 ndarray contraining the labels
    """
    train_all = np.loadtxt('data/smallTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt('data/smallValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def load_data_medium():
    """ Load medium training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An N_train-x-M ndarray containing the training data (N_train examples, M features each)
        y_train: An N_train-x-1 ndarray contraining the labels
        X_val: An N_val-x-M ndarray containing the training data (N_val examples, M features each)
        y_val: An N_val-x-1 ndarray contraining the labels
    """
    train_all = np.loadtxt('data/mediumTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt('data/mediumValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def load_data_large():
    """ Load large training and validation dataset

        Returns a tuple of length 4 with the following objects:
        X_train: An N_train-x-M ndarray containing the training data (N_train examples, M features each)
        y_train: An N_train-x-1 ndarray contraining the labels
        X_val: An N_val-x-M ndarray containing the training data (N_val examples, M features each)
        y_val: An N_val-x-1 ndarray contraining the labels
    """
    train_all = np.loadtxt('data/largeTrain.csv', dtype=int, delimiter=',')
    valid_all = np.loadtxt('data/largeValidation.csv', dtype=int, delimiter=',')

    X_train = train_all[:, 1:]
    y_train = train_all[:, 0]
    X_val = valid_all[:, 1:]
    y_val = valid_all[:, 0]

    return (X_train, y_train, X_val, y_val)


def linearForward(input, p):
    """
    :param input: input vector (column vector) WITH bias feature added
    :param p: parameter matrix (alpha/beta) WITH bias parameter added
    :return: output vector
    """
    
    return p.dot(input)


def sigmoidForward(a):
    """
    :param a: input vector WITH bias feature added
    """
    return np.array(list(map(lambda x : 1/(1+np.exp(-x)),a)))


def softmaxForward(b):
    """
    :param b: input vector WITH bias feature added
    """
    tmp = np.array(list(map(lambda x : np.exp(x),b)))
    return tmp/sum(tmp)


def crossEntropyForward(hot_y, y_hat):
    """
    :param hot_y: 1-hot vector for true label
    :param y_hat: vector of probabilistic distribution for predicted label
    :return: float
    """
    return np.sum(hot_y*(np.log(y_hat)))


def NNForward(x, y, alpha, beta):
    """
    :param x: input data (column vector) WITH bias feature added
    :param y: input (true) labels
    :param alpha: alpha WITH bias parameter added
    :param beta: alpha WITH bias parameter added
    :return: all intermediate quantities x, a, z, b, y, J #refer to writeup for details
    TIP: Check on your dimensions. Did you make sure all bias features are added?
    """
    a = linearForward(x,alpha)
    z = sigmoidForward(a)
    b = linearForward(z,beta)
    y_hat = softmaxForward(b)
    J = crossEntropyForward(y,y_hat)
    return x, a, z, b, y_hat, J


def softmaxBackward(hot_y, y_hat):
    """
    :param hot_y: 1-hot vector for true label
    :param y_hat: vector of probabilistic distribution for predicted label
    """
    return y_hat - hot_y
    


def linearBackward(prev, p, grad_curr):
    """
    :param prev: previous layer WITH bias feature
    :param p: parameter matrix (alpha/beta) WITH bias parameter
    :param grad_curr: gradients for current layer
    :return:
        - grad_param: gradients for parameter matrix (alpha/beta)
        - grad_prevl: gradients for previous layer
    TIP: Check your dimensions.
    """
    grad_prevl = p.T.dot(grad_curr) # p = Kx(D+1), grad_curr = Kx1 -> grad_prevl = (D+1)x1
    grad_param = grad_curr.dot(prev.T) # prev = (D+1)x1, grad_curr = Kx1 -> grad_param = Kx(D+1)
    return grad_param, grad_prevl
    


def sigmoidBackward(curr, grad_curr):
    """
    :param curr: current layer WITH bias feature
    :param grad_curr: gradients for current layer
    :return: grad_prevl: gradients for previous layer
    TIP: Check your dimensions
    """
    return np.r_([np.zeros(curr.shape[0])],[np.diag(curr*(1-curr))]) * grad_curr # grad_curr = (D+1)x1 -> grad_prevl = Dx1 -----> need Dx(D+1) matrix


def NNBackward(x, y, alpha, beta, z, y_hat):
    """
    :param x: input data (column vector) WITH bias feature added
    :param y: input (true) labels
    :param alpha: alpha WITH bias parameter added
    :param beta: alpha WITH bias parameter added
    :param z: z as per writeup
    :param y_hat: vector of probabilistic distribution for predicted label
    :return:
        - grad_alpha: gradients for alpha
        - grad_beta: gradients for beta
        - g_b: gradients for layer b (softmaxBackward)
        - g_z: gradients for layer z (linearBackward)
        - g_a: gradients for layer a (sigmoidBackward)
    """
    g_b = softmaxBackward(y,y_hat)
    g_beta,g_z = linearBackward(z,beta,g_b)
    g_a = sigmoidBackward(z,g_z)
    g_alpha,_ = linearBackward(x,alpha,g_a)
    
    return g_alpha, g_beta, g_b, g_z, g_a



def SGD(tr_x, tr_y, valid_x, valid_y, hidden_units, num_epoch, init_flag, learning_rate):
    """
    :param tr_x: Training data input (size N_train x M)
    :param tr_y: Training labels (size N_train x 1)
    :param tst_x: Validation data input (size N_valid x M)
    :param tst_y: Validation labels (size N_valid x 1)
    :param hidden_units: Number of hidden units
    :param num_epoch: Number of epochs
    :param init_flag:
        - True: Initialize weights to random values in Uniform[-0.1, 0.1], bias to 0
        - False: Initialize weights and bias to 0
    :param learning_rate: Learning rate
    :return:
        - alpha weights
        - beta weights
        - train_entropy (length num_epochs): mean cross-entropy loss for training data for each epoch
        - valid_entropy (length num_epochs): mean cross-entropy loss for validation data for each epoch
    """
    N,M = tr_x.shape
    if init_flag:
        alpha = np.r_(np.zeros(hidden_units).T,np.random.uniform(-0.1,0.1,size = (hidden_units,M)))
        beta = np.r_(np.zeros(10).T,np.random.unifrom(-0.1,0.1,size = (10,hidden_units)))
    else:
        alpha = np.zeros((hidden_units,M+1))
        beta = np.zeros(10,hidden_units+1)
    cr_train_lst = []
    cr_valid_lst = []
    for e in range(1,num_epoch+1):
        for i in range(0,N):
            x = tr_x[i]
            y = tr_y[i]
            val_x = valid_x[i]
            val_y = valid_y[i]
            x,a,b,z,y_hat,J = NNForward(x,y,alpha,beta)
            _,_,_,_,_,J_valid = NNForward(val_x,val_y,alpha,beta)
            
            g_alpha, g_beta,_,_,_ = NNBackward(x,y,alpha,beta,z,y_hat)
            alpha = alpha - learning_rate*g_alpha
            beta = beta - learning_rate*g_beta
            cr_train_lst.append(J/N)
            cr_valid_lst.append(J_valid/N)
            
    return alpha, beta, cr_train_lst, cr_valid_lst


def prediction(tr_x, tr_y, valid_x, valid_y, tr_alpha, tr_beta):
    """
    :param tr_x: Training data input (size N_train x M)
    :param tr_y: Training labels (size N_train x 1)
    :param valid_x: Validation data input (size N_valid x M)
    :param valid_y: Validation labels (size N-valid x 1)
    :param tr_alpha: Alpha weights WITH bias
    :param tr_beta: Beta weights WITH bias
    :return:
        - train_error: training error rate (float)
        - valid_error: validation error rate (float)
        - y_hat_train: predicted labels for training data
        - y_hat_valid: predicted labels for validation data
    """
    N,M = tr_x.shape
    train_error,valid_error = 0
    y_hat_train,y_hat_valid = []
    
    for i in range(N):
        x = tr_x[i]
        y = tr_y[i]
        val_x = valid_x[i]
        val_y = valid_y[i]
        
        y_hat = NNForward(x,y,tr_alpha,tr_beta)
        y_hat_val = NNForward(val_x,val_y,tr_alpha,tr_beta)
        
        l = np.argmax(y_hat)
        l_val = np.argmax(y_hat_val)
        
        y_hat_train.append(l)
        y_hat_valid.append(l_val)
        
        if l != y:
            train_error +=1
        if l_val != val_y:
            valid_error+=1
    return train_error/N, valid_error/N, y_hat_train, y_hat_valid

### FEEL FREE TO WRITE ANY HELPER FUNCTIONS

def train_and_valid(X_train, y_train, X_val, y_val, num_epoch, num_hidden, init_rand, learning_rate):
    """ Main function to train and validate your neural network implementation.

        X_train: Training input in N_train-x-M numpy nd array. Each value is binary, in {0,1}.
        y_train: Training labels in N_train-x-1 numpy nd array. Each value is in {0,1,...,K-1},
            where K is the number of classes.
        X_val: Validation input in N_val-x-M numpy nd array. Each value is binary, in {0,1}.
        y_val: Validation labels in N_val-x-1 numpy nd array. Each value is in {0,1,...,K-1},
            where K is the number of classes.
        num_epoch: Positive integer representing the number of epochs to train (i.e. number of
            loops through the training data).
        num_hidden: Positive integer representing the number of hidden units.
        init_flag: Boolean value of True/False
        - True: Initialize weights to random values in Uniform[-0.1, 0.1], bias to 0
        - False: Initialize weights and bias to 0
        learning_rate: Float value specifying the learning rate for SGD.

        RETURNS: a tuple of the following six objects, in order:
        loss_per_epoch_train (length num_epochs): A list of float values containing the mean cross entropy on training data after each SGD epoch
        loss_per_epoch_val (length num_epochs): A list of float values containing the mean cross entropy on validation data after each SGD epoch
        err_train: Float value containing the training error after training (equivalent to 1.0 - accuracy rate)
        err_val: Float value containing the validation error after training (equivalent to 1.0 - accuracy rate)
        y_hat_train: A list of integers representing the predicted labels for training data
        y_hat_val: A list of integers representing the predicted labels for validation data
    """
    ### YOUR CODE HERE
    loss_per_epoch_train = []
    loss_per_epoch_val = []
    
    err_train = None
    err_val = None
    y_hat_train = None
    y_hat_val = None
    
    
    alpha, beta, cr_train_lst, cr_valid_lst = SGD(X_train,y_train,X_val,y_val, num_hidden, num_epoch, init_rand, learning_rate)
    
    err_train,err_val, y_hat_train , y_hat_val = prediction(X_train,y_train,X_val,y_val,alpha, beta)
    
    loss_per_epoch_train = cr_train_lst
    loss_per_epoch_val = cr_valid_lst

    return (loss_per_epoch_train, loss_per_epoch_val,
            err_train, err_val, y_hat_train, y_hat_val)
    
X_train, y_train, X_val, y_val = load_data_small()
loss_train, loss_val, error_train, error_val, y_hat_train, y_hat_val = train_and_valid(X_train, y_train, X_val, y_val, num_epoch=100, num_hidden=10, init_rand=True, learning_rate=0.01)

# Print the results
print("Training Loss: ", loss_train[-1])
print("Validation Loss: ", loss_val[-1])
print("Training Error: ", error_train)
print("Validation Error: ", error_val)