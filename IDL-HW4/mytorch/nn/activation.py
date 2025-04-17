import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        
        # Shift input for numerical stability: subtract max along softmax dim
        Z_shifted = Z - np.max(Z, axis=self.dim, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        sum_exp_Z = np.sum(exp_Z, axis=self.dim, keepdims=True)
    
        # Compute softmax probabilities
        self.A = exp_Z / sum_exp_Z

        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        
        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = shape[self.dim]
           
        # Reshape input to 2D
        if len(shape) > 2:
            A_moved = np.moveaxis(self.A, self.dim, -1)
            dLdA_moved = np.moveaxis(dLdA, self.dim, -1)

            N = np.prod(A_moved.shape[:-1])
            A_flat = A_moved.reshape(N, C)
            dLdA_flat = dLdA_moved.reshape(N, C)

            dLdZ_flat = np.zeros_like(A_flat)

            for i in range(N):
                a = A_flat[i]
                J = np.diag(a) - np.outer(a, a)
                dLdZ_flat[i] = dLdA_flat[i] @ J

            # Reshape and move axis back to original
            dLdZ_moved = dLdZ_flat.reshape(A_moved.shape)
            dLdZ = np.moveaxis(dLdZ_moved, -1, self.dim)
        else:
            N = shape[0]
            dLdZ = np.zeros_like(self.A)
            for i in range(N):
                a = self.A[i]
                J = np.diag(a) - np.outer(a, a)
                dLdZ[i] = dLdA[i] @ J

        return dLdZ
 

    