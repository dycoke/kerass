import numpy as np

class DenseLayer:
    def __init__(self, neurons) -> None:
        self.neurons = neurons
    
    def relu(self, inputs):
        # ReLU activation

        return np.maximum(0, inputs)
    
    def d_relu(self, dA, Z):
        # derivative of ReLU
        pass
    
    def softmax(self, inputs):
        # softmax activation

        scores = np.exp(inputs)
        ret = scores / np.sum(scores, axis = 1, keepdims=True)
    
    def forward(self, inputs, weights, bias, activation):
        # single layer forward propogation

        Z_curr = np.dot(inputs, weights.T)
        if activation == 'relu':
            A_curr = self.relu(Z_curr)
        else:
            A_curr = self.softmax(Z_curr)
        
        return A_curr, Z_curr
        

    
    def backward(self, dA_curr, W_curr, Z_curr, A_prev, activation):
        # single layer backwards propogation
        pass

class Network:
    def __init__(self) -> None:
        self.network = [] # layers
        self.architecture = [] # mapping input neurons to output neurons
        self.params = [] # weights (W), biases (b)
        self.memory = [] # (Z), activation (A)
        self.gradients = [] # dW, db
    
    def add(self, layer):
        # add a layer to the network
        self.network.append(layer)

    def _compile(self, data):
        # compile network
        for idx, layer in enumerate(self.network):
            if idx == 0:
                # this is the input layer
                self.architecture.append({
                    'input_shape': data.shape[1],
                    'output_shape': self.network[idx].neurons,
                    'activation': 'relu'
                })
            elif idx == len(self.network) - 1:
                # last layer
                self.architecture.append({
                    'input_shape': self.network[idx-1].neurons,
                    'output_shape': self.network[idx].neurons,
                    'activation': 'softmax'
                })
            else:
                # hidden layers
                self.architecture.append({
                    'input_shape': self.network[idx-1].neurons,
                    'output_shape': self.network[idx].neurons,
                    'activation': 'relu'
                })
        
        return self
    
    def _init_weights(self, data):
        # initialize the prarmeters for the model

        self._compile(data)

        np.random.seed(15257)
        for i in range(len(self.architecture)):
            self.params.append({
                'W': np.random.uniform(low = -1, high = 1,
                size = (self.architecture[i]['output_shape'], 
                    self.architecture[i]['input_shape'])),
                'b': np.zeros((1, self.architecture[i]['output_shape']))
            })
        
        return self
    
    def _forwardprop(self, data):
        # performs a forward pass

        A_curr = data
        for i in range(len(self.params)):
            A_prev = A_curr
            A_curr, Z_curr = self.network[i].forward(
                A_prev,
                self.params[i]['W'],
                self.params[i]['b'],
                self.architecture[i]['activation']
            )
            self.memory.append({'inputs':A_prev, 'Z':Z_curr})
    
    def _backprop(self, predicted, actual):
        # performs a backwards propogation
        pass
    
    def _update(self, lr = 0.01):
        # update the prarmeters using lr as the constant factor on the gradient
        pass
    
    def _calc_loss(self, predicted, actual):
        # calculate and return loss using cross-entropy
        pass
    
    def train(self, X_train, y_train, epochs):
        # train the model using stochastic gradient descent
        pass