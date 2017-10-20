"""
Neural Network from scratch. 
A simple Neural Network calss. 
License MIT, all rights reserved jerry liu @twairball
"""
import numpy as np

# sigmoid and sigmoid derivative functions
def sigmoid(x):
    x = np.clip(x, -500, 500) # avoid overflow
    return 1 / ( 1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

class NN:
    """
    A simple Neural Network class. 
    params:
        lr: learning rate, regularizer term for weights W_n update
        during gradient descent update. default=0.0001. 
    
    methods:
        train: trains neural network for number of iterations
        predit: predict outputs and evaluates loss. 
    """
    def __init__(self, lr=0.0001):
        self.weights = self.init_weights()
        self.biases = self.init_biases()
        self.lr = lr

    def init_weights(self):
        # initialize weights with mean 0
        # network is 3, 4, 1
        W_1 = 2 * np.random.random((3, 4)) - 1 # shape: (3,4)
        W_2 = 2 * np.random.random((4, 1)) - 1 # shape: (4,1)
        return [W_1, W_2]

    def init_biases(self):
        # initialize bias as zeros
        # bias shape is same as dimension of W_1, W_2
        b_1 = np.zeros(4) # shape: (4,1)
        b_2 = np.zeros(1) # shape: (1,1)
        return [b_1, b_2]

    def forward_pass(self, inputs):
        W_1, W_2 = self.weights
        b_1, b_2 = self.biases

        h0 = inputs
        h1 = sigmoid(h0.dot(W_1) + b_1)
        h2 = sigmoid(h1.dot(W_2) + b_2)
        return h0, h1, h2

    def back_prop(self, targets, activations):
        h0, h1, h2 = activations
        W_1, W_2 = self.weights
        b_1, b_2 = self.biases

        h2_loss = targets - h2 # shape (4,1)
        h2_delta = h2_loss * sigmoid_deriv(h2) # shape (4,1)

        h1_loss = h2_delta.dot(W_2.T)
        h1_delta = h1_loss * sigmoid_deriv(h1) # shape (4,4)

        # dW calculated as dot(input, delta)
        # added regularization term 
        dW_2 = h1.T.dot(h2_delta) + self.lr * W_2
        dW_1 = h0.T.dot(h1_delta) + self.lr * W_1

        # db calculated as sum of delta
        db_2 = np.sum(h2_delta, axis=0)
        db_1 = np.sum(h1_delta, axis=0)

        # gradient descent update
        self.weights = [W_1 + dW_1, W_2 + dW_2]
        self.biases = [b_1 + db_1, b_2 + db_2]
    
    def predict(self, inputs, targets):
        """ Returns predictions given inputs. 
        Also computes 
        """
        activations = self.forward_pass(inputs)
        preds = activations[-1]
        loss = np.mean(np.abs(targets - preds))
        return preds, loss
    
    def train(self, inputs, targets, n_iters = 60000, print_every=5000):
        """ Train NN to fit targets given inputs. 
        Inputs and outputs should be a 2-D matrix. 
        """     
        for i in range(int(n_iters)):
            activations = self.forward_pass(inputs)
            self.back_prop(targets, activations)
            
            if (i % print_every) == 0:
                preds = activations[-1]
                loss = np.mean(np.abs(targets - preds))
                print("[%d] loss: %f" % (i, loss))

def main():
    # input and outputs
    X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]]) # shape: (4, 3)
    Y = np.array([[0,1,1,0]]).T # shape: (4,1)

    # create a neural net and train it
    nn = NN()
    nn.train(X, Y)

    # evaluate predicted outputs
    preds, loss = nn.predict(X, Y)
    print("predicted output: %s" % preds)        
    print("loss: %f" % loss)


if __name__ == "__main__":
    main()
