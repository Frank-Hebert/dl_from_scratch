class Network:
    def __init__(self):
        self.network = [] # Layers
        self.architecture = [] # map input neurons -> output neurons
        self.params = [] # W, b
        self.memory = [] # Z, A
        self.gradients = [] # dW, db

    def add(self, layer):
        """
        Adding layer to the network
        """
        self.network.append(layer)

    def _compile(self, data):
        """
        Initialize model architecture
        """
        raise NotImplementedError
    
    def _init_weights(self, data):
        """
        Initialize the weights/parameters
        """
        raise NotImplementedError
    
    def _forwardprop(self, data):
        """
        Do one full forward pass
        """
        raise NotImplementedError
    
    def _backprop(self, data):
        """
        Do one full backward pass
        """
        raise NotImplementedError
            
    def _update(self, lr=0.01):
        """
        Update the model weights -> lr * gradients
        """
        raise NotImplementedError

    def _get_accuracy(self, predicted, actual):
        """
        Calculate accuracy after each iteration
        """
        raise NotImplementedError

    def _calculate_loss(self, predicted, actual):
        """
        Calculate loss after each iteration
        """

        raise NotImplementedError

    def train(self, X_train, y_train, epochs):
        """
        Train model using SGD
        """
        raise NotImplementedError
