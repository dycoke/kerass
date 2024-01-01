import numpy as np

class DenseLayer:
    def __init__(self, neurons) -> None:
        self.neurons = neurons
    
    def relu(self, inputs):
        # ReLU activation
        return np.maximum(0, inputs)
    
    def d_relu(self, dA, Z):
        # derivative of ReLU
        dZ = np.array(dA, copy = True)
        dZ[Z <= 0] = 0
        return dZ
    
    def softmax(self, inputs):
        # softmax activation

        scores = np.exp(inputs - np.max(inputs, axis=1)[:, np.newaxis])
        ret = scores / np.sum(scores, axis = 1, keepdims=True)
        return ret
    
    def forward(self, inputs, weights, bias, activation):
        # single layer forward propogation

        Z_curr = np.dot(inputs, weights.T) + bias
        if activation == 'relu':
            A_curr = self.relu(Z_curr)
        elif activation == 'softmax':
            A_curr = self.softmax(Z_curr)
        
        return A_curr, Z_curr
        

    
    def backward(self, dA_curr, W_curr, Z_curr, A_prev, activation):
        # single layer backwards propogation
        if activation == 'softmax':
            dW = np.dot(A_prev.T, dA_curr)
            db = np.sum(dA_curr, axis=0, keepdims=True)
            dA = np.dot(dA_curr, W_curr) 
        else:
            dZ = self.d_relu(dA_curr, Z_curr)
            dW = np.dot(A_prev.T, dZ)
            db = np.sum(dZ, axis=0, keepdims=True)
            dA = np.dot(dZ, W_curr)
        
        return dA, dW, db

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

        np.random.seed(1352)
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
            self.memory.append({'inputs': A_prev, 'Z': Z_curr})
        
        return A_curr
    
    def _backprop(self, predicted, actual):
        # performs a backwards propogation
        num_samples = len(actual)

        dscores = predicted
        dscores[range(num_samples), actual] -= 1
        dscores /= num_samples

        dA_prev = dscores
        for idx, layer in reversed(list(enumerate(self.network))):
            dA_curr = dA_prev

            A_prev = self.memory[idx]['inputs']
            Z_prev = self.memory[idx]['Z']
            W_prev = self.params[idx]['W']

            activation = self.architecture[idx]['activation']

            dA_prev, dW_curr, db_curr = layer.backward(dA_curr, W_prev, Z_prev, A_prev, activation)

            self.gradients.append({'dW': dW_curr, 'db': db_curr})

    
    def _update(self, lr = 0.01):
        # update the prarmeters using lr as the constant factor on the gradient
        for idx, layer in enumerate(self.network):
            self.params[idx]['W'] -= lr * list(reversed(self.gradients))[idx]['dW'].T  
            self.params[idx]['b'] -= lr * list(reversed(self.gradients))[idx]['db']

    def _calc_accuracy(self, predicted, actual):
        #calculate accuracy
        return np.mean(np.argmax(predicted, axis = 1) == actual)
    
    def _calc_loss(self, predicted, actual):
        # calculate and return loss using cross-entropy
        samples = len(actual)

        correct_logprobs = 0
        for c in predicted[range(samples), actual]:
            if c <= 0:
                c = 0.01
            #add a small episilon to prevent bad things
            correct_logprobs += -np.log(c)
        loss = correct_logprobs / samples

        return loss
    
    def train(self, X_train, y_train, epochs):
        # train the model using stochastic gradient descent
        self.loss = []
        self.accuracy = []

        self._init_weights(X_train)

        for i in range(1, epochs + 1):
            yhat = self._forwardprop(X_train)
            self.accuracy.append(self._calc_accuracy(yhat, y_train))
            self.loss.append(self._calc_loss(yhat, y_train))

            self._backprop(yhat, y_train)

            self._update()

            if i % 20 == 0:
                print(f'EPOCH: {i}, ACCURACY: {self.accuracy[-1]}, LOSS: {self.loss[-1]}')