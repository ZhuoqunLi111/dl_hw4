import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # TODO: Implement forward pass
        
        self.A = A
        self.input_shape = A.shape

        # Flatten the input to 2D shape (batch_size, in_features)
        A_flat = A.reshape(-1, self.W.shape[1])

        # Compute affine transformation Z = A @ W.T + b
        Z_flat = A_flat @ self.W.T + self.b

        # Reshape back to original shape with last dim = out_features
        Z = Z_flat.reshape(*self.input_shape[:-1], self.W.shape[0])
        
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass

        # Reshape dLdZ to 2D: (batch_size, out_features)
        dLdZ_flat = dLdZ.reshape(-1, self.W.shape[0])
    
        # Reshape saved input A to 2D: (batch_size, in_features)
        A_flat = self.A.reshape(-1, self.W.shape[1])

        # Compute gradients (refer to the equations in the writeup)
        self.dLdA = dLdZ_flat @ self.W
        self.dLdW = dLdZ_flat.T @ A_flat
        self.dLdb = dLdZ_flat.sum(axis=0)
        self.dLdA = self.dLdA.reshape(*self.input_shape)
        
        # Return gradient of loss wrt input
        return self.dLdA
