"""
Neural Network from scratch. 
License MIT, all rights reserved jerry liu @twairball
"""
import numpy as np

# input and outputs
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]]) # shape: (4, 3)
Y = np.array([[0,1,1,0]]).T # shape: (4,1)

# regularization term, or learning rate
lr = 0.0001

# number of training iterations
n_iters = 60000

def initialize():
    # initialize weights with mean 0
    # network is 3, 4, 1
    W_1 = 2 * np.random.random((3, 4)) - 1 # shape: (3,4)
    W_2 = 2 * np.random.random((4, 1)) - 1 # shape: (4,1)

    # initialize bias as zeros
    # bias shape is same as dimension of W_1, W_2
    b_1 = np.zeros(4) # shape: (4,1)
    b_2 = np.zeros(1) # shape: (1,1)

    return [W_1, W_2], [b_1, b_2]

# sigmoid and sigmoid derivative functions
def sigmoid(x):
    x = np.clip(x, -500, 500) # avoid overflow
    return 1 / ( 1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

def forward_pass(inputs, weights, biases):
    W_1, W_2 = weights
    b_1, b_2 = biases

    h0 = inputs
    h1 = sigmoid(h0.dot(W_1) + b_1)
    h2 = sigmoid(h1.dot(W_2) + b_2)
    return h0, h1, h2

def back_prop(activations, weights, biases):
    h0, h1, h2 = activations
    W_1, W_2 = weights
    b_1, b_2 = biases

    h2_loss = Y - h2 # shape (4,1)
    h2_delta = h2_loss * sigmoid_deriv(h2) # shape (4,1)

    h1_loss = h2_delta.dot(W_2.T)
    h1_delta = h1_loss * sigmoid_deriv(h1) # shape (4,4)

    # dW calculated as dot(input, delta)
    # added regularization term 
    dW_2 = h1.T.dot(h2_delta) + lr * W_2
    dW_1 = h0.T.dot(h1_delta) + lr * W_1

    # db calculated as sum of delta
    db_2 = np.sum(h2_delta, axis=0)
    db_1 = np.sum(h1_delta, axis=0)

    # gradient descent update
    weights = [W_1 + dW_1, W_2 + dW_2]
    biases = [b_1 + db_1, b_2 + db_2]

    return weights, biases

def predict(weights, biases):
    h0, h1, h2 = forward_pass(X, weights, biases)
    print("predicted output: %s" % h2)
    loss = np.mean(np.abs(Y-h2))
    print("loss: %f" % loss)

def main():
    weights, biases = initialize()

    # train for N iterations
    for i in range(int(n_iters)):
        activations = forward_pass(X, weights, biases)
        weights, biases = back_prop(activations, weights, biases)

        if (i % 5000) == 0:
            preds = activations[-1]
            loss = np.mean(np.abs(Y - preds))
            print("[%d] loss: %f" % (i, loss))
    
    # predict our output and check results
    predict(weights, biases)

if __name__ == "__main__":
    main()
