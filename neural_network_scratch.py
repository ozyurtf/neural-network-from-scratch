import torch

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        self.a0 = x
        self.s1 = torch.matmul(x, self.parameters['W1'].T) + self.parameters['b1']

        if self.f_function == "relu": 
            self.a1 = torch.max(torch.zeros(self.s1.shape), self.s1)

        elif self.f_function == "sigmoid":
            self.a1 = 1 / (1 + torch.exp(-self.s1))

        elif self.f_function == "identity": 
            self.a1 = self.s1

        self.s2 = torch.matmul(self.a1, self.parameters['W2'].T) + self.parameters['b2'] 

        if self.g_function == "relu": 
            self.a2 = torch.max(torch.zeros(self.s2.shape), self.s2)

        elif self.g_function == "sigmoid":
            self.a2 = 1 / (1 + torch.exp(-self.s2))

        elif self.g_function == "identity": 
            self.a2 = self.s2

        self.y_hat = self.a2
        return self.y_hat
    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """ 
        ###########

        dJdW1s = []
        dJdW2s = []
        
        dJdb1s = []
        dJdb2s = []

        ###########

        # The architecture of the neural network is like this: 
        # a0 = x
        # s1 = W1.x + b1
        # a1 = f(s1)
        # s2 = W2.a1 + b2 
        # a2 = g(s2)
        # y_pred = a2

        # The codes below calculate the graidents for each data point in a for loop. If we have a 10 data points, for example, 
        # the codes below calculates the gradients for each one of them, and gathers them in dJdW1s, dJdW2s, dJdb1s, and dJdb2s. 
        # At the end the average of these gradients is taken and this average is used to update the W1, W2, b1, and b2.
        
        for val in range(dJdy_hat.shape[0]):

        ###########

            # The codes below take the partial derivative of the a1 with respect to s1. 
            # The partial derivative is different depending on the activation function. 
            if self.f_function == "relu": 
                da1_ds1 = torch.diag_embed(torch.where(self.s1[val] < 0, torch.tensor(1), torch.tensor(0))).float()

            elif self.f_function == "sigmoid": 
                da1_ds1 = torch.diag_embed(torch.sigmoid(self.s1[val]) * (1 - torch.sigmoid(self.s1[val]))).float()

            elif self.f_function == "identity": 
                da1_ds1 = torch.eye(self.s1.shape[1]).float()

            ###########

            if self.g_function == "relu": 
                da2_ds2 = torch.diag_embed(torch.where(self.s2[val] < 0, torch.tensor(1), torch.tensor(0))).float()

            elif self.g_function == "sigmoid": 
                da2_ds2 = torch.diag_embed(torch.sigmoid(self.s2[val]) * (1 - torch.sigmoid(self.s2[val]))).float()

            elif self.g_function == "identity": 
                da2_ds2 = torch.eye(self.s2.shape[1]).float()

            ###########   
            
            # In our model architecture, s1 = W1.x + b1.
            # The partial derivative of s1 with respect to W1 is calculated in the code below.
            # If our input has 2 features, for example, and the dimension of the W1 is 20x2,
            # our ds1_dW1 will be like this:
        
            # | |x1, x2| |
            # | |0,  0 | |
            # | |0,  0 | |
            # | |0,  0 | |
            # | |....  | |
            # |          |
            # | |0,  0 | |
            # | |x1, x2| |
            # | |0,  0 | |
            # | |0,  0 | |
            # | |....  | |
            # |          |
            # | |0,  0 | |
            # | |0,  0 | |
            # | |x1, x2| |
            # | |0,  0 | |
            # | |....  | |
            # |          |
            # |   ...    |
            # |          |
            # |          |  

            # And it's dimension will be 20 x (20 x 2)

            a0_input = torch.tensor(self.a0[val])
            ds1_dW1_matrices = []
            for i in range(da1_ds1[0].shape[-1]):
                W1_zero_matrix = torch.zeros((self.parameters['W1'].shape[0], 
                                              self.parameters['W1'].shape[1]))
                W1_zero_matrix[i] = a0_input
                ds1_dW1_matrices.append(W1_zero_matrix)
            ds1_dW1 = torch.stack(ds1_dW1_matrices, dim=0)
            
            ###########

            # In our model architecture, s2 = W2.a1 + b2.
            # The partial derivative of s2 with respect to W2 (ds2_dW2) is calculated in the code below 
            # in a similar way to the calculating ds1_dW1

            a1_input = torch.tensor(self.a1[0])
            ds2_dW2_matrices = []

            for i in range(da2_ds2[0].shape[-1]):
                W2_zero_matrix = torch.zeros((self.parameters['W2'].shape[0], 
                                              self.parameters['W2'].shape[1]))
                W2_zero_matrix[i] = a1_input
                ds2_dW2_matrices.append(W2_zero_matrix)

            ds2_dW2 = torch.stack(ds2_dW2_matrices, dim=0)  
            
            ###########
            
            # Calculating the partial derivative of s2 with respect to b2 
            ds2_db2 = torch.ones(self.s2[val].shape[0])

            # Calculating the partial derivative of s1 with respect to b1
            ds1_db1 = torch.ones(self.s1[val].shape[0])
            
            ###########  
            
            # Calculating the partial derivative of s2 with respect to a1
            ds2_da1 = self.parameters['W2']

            ###########
            
            # Chain rule
            dJ_ds2 = torch.matmul(dJdy_hat[val].view(-1), da2_ds2)
            dJ_da1 = torch.matmul(dJ_ds2, ds2_da1)
            dJ_ds1 = torch.matmul(dJ_da1, da1_ds1)  
            
            dJdW1 = torch.matmul(dJ_ds1, ds1_dW1)
            dJdW2 = torch.matmul(dJ_ds2, ds2_dW2)
            
            dJdb1 = torch.matmul(dJ_ds1, ds1_db1)
            dJdb2 = torch.matmul(dJ_ds2, ds2_db2)  

            ########### 

            dJdW1s.append(dJdW1)
            dJdW2s.append(dJdW2)
            
            dJdb1s.append(dJdb1)
            dJdb2s.append(dJdb2) 

            ###########

        self.grads['dJdW2'] = torch.stack(dJdW2s, dim=0).mean(dim=0)
        self.grads['dJdW1'] = torch.stack(dJdW1s, dim=0).mean(dim=0)
        self.grads['dJdb2'] = torch.stack(dJdb2s, dim=0).mean(dim=0)
        self.grads['dJdb1'] = torch.stack(dJdb1s, dim=0).mean(dim=0)              
            
        # Updating parameters.   
        self.parameters['W2'] = self.parameters['W2'] - 0.001 * self.grads['dJdW2']
        self.parameters['W1'] = self.parameters['W1'] - 0.001 * self.grads['dJdW1']  
        self.parameters['b2'] = self.parameters['b2'] - 0.001 * self.grads['dJdb2']
        self.parameters['b1'] = self.parameters['b1'] - 0.001 * self.grads['dJdb1']
        pass
    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    
    mse_loss_values = []
    for (y_i, y_hat_i) in zip(y, y_hat):
        y_hat_i_sigmoid = torch.sigmoid(y_hat_i)
        mse_loss_i = (y_i - y_hat_i)**2
        mse_loss_values.append(mse_loss_i)
    
    mse_loss_values = torch.stack(mse_loss_values)
    mse_loss = torch.mean(mse_loss_values)

    # Calculating the partial derivative of the MSE Loss function with respect to y_hat.
    dJdy_hat = 2 * (y_hat - y)
    
    return mse_loss, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """

    bce_loss_values = []
    for (y_i, y_hat_i) in zip(y, y_hat):
        y_hat_i_sigmoid = torch.sigmoid(y_hat_i)
        bce_loss_i = -torch.mean((y_i * torch.log(y_hat_i_sigmoid) + (1 - y_i) * torch.log(1 - y_hat_i_sigmoid)))
        bce_loss_values.append(bce_loss_i)
    
    bce_loss_values = torch.stack(bce_loss_values)
    bce_loss = torch.mean(bce_loss_values)

    # Calculating the partial derivative of the BCE Loss function with respect to y_hat.
    dJdy_hat = (- 1 / len(y))*(1/torch.log(torch.tensor(2.0)))*(y/y_hat - (1-y)/(1-y_hat))
    
    return bce_loss, dJdy_hat
