"""
    - XOR truth table using sgd back-propagation
    Author:
        Michael Magaisa
    Date:
        8.09.2020
"""
#use keras only to show the neural network model
import keras
from keras.models import Sequential
from keras.layers import Dense
from ann_visualizer.visualize import ann_viz;

#for matrices and vectorization
import numpy as np

class network_model(object):
    """
        - A model of the network
    """
    def __init__(self,input_layer,hidden_layer,output_layer,
                        hidden_activation,output_activation):
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
    
    def view_pdf(self):
        """
            - Used to visualize the neural network model 
        """
        model = Sequential()
        model.add(Dense(self.hidden_layer, input_dim = self.input_layer,
                        activation = self.hidden_activation))
        model.add(Dense(self.output_layer,
                         activation = self.hidden_activation))
        ann_viz(model, title="Back propagation - XOR truth table")

    def sigmoid(self,x):
        self.x = x
        return (1/(1 + np.exp(-x)))

    def sgd_back_propagation(self,W1,W2,X,Y):
        """
            - Implimentation of the algorithm
        """
        alpha = 0.9
        self.W1 = W1 #the initial weight matrix for input 1
        self.W2 = W2 #the initial weight matrix for input 2
        self.X = X #the input vactor
        self.Y = Y #the target vector

        N = 4 #the rows of the truth table
        for i in range(N):
            x = self.X[i,:] #the input row
            x = np.transpose(x) #make it a single column vector
            y = self.Y[i] #the correct output

            #output layer weighted sum and activation output
            #wighted sum of hidden layer
            v_hidden = self.W1 @ x 
            #the output from hidden layer. Will be fed as input to next layer
            y_hidden = self.sigmoid(v_hidden)

            #output layer weighted sum and activation output 
            v_out = self.W2 @ y_hidden
            y_out = self.sigmoid(v_out)

            #error calculation, back propagation part1
            out_error = y - y_out
            out_delta = y_out*(1-y_out)*out_error

            hidden_error = np.transpose(self.W2) @ out_delta
            hidden_delta = np.multiply(y_hidden,(1-y_hidden))
            hidden_delta = np.multiply(hidden_delta,hidden_error) #alternative elementwise numpy multiplication 

            #changes in weight matrix feeding first layer
            dW1 = alpha*hidden_delta*x[:,np.newaxis] #broadcast to allow the dimensions to match
            self.W1 = self.W1 + np.transpose(dW1) #transpose to make the addition possible

            #changes in the weight matrix feeding the output layer
            dW2 = alpha*out_delta*y_hidden[:,np.newaxis]
            self.W2 = self.W2 + np.transpose(dW2)

        return self.W1 , self.W2

    def impliment_model(self):
        """
            Train and test the model
        """
        #input data array
        X = np.array([[0, 0, 1],
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]])

        #output data column vector
        Y = np.array([0,1,1,0])
        Y.shape = (4,1) #make it a 4*1 column vector

        #initialize the weights
        W1 = 2*np.random.uniform(0,1,(4,3)) -1
        W2 = 2*np.random.uniform(0,1,(1,4)) -1

        #train the model
        for epoch in range(100000): #10 000 iterations can stil produce good enough results
            W1 , W2 = self.sgd_back_propagation(W1,W2,X,Y)

        #inference
        N = 4 #number of truth table rows
        for i in range(N):
            x = X[i,:]
            x = np.transpose(x)
            hidden_sum = W1 @ x
            hidden_out = self.sigmoid(hidden_sum)
            out_sum = W2 @ hidden_out
            output = self.sigmoid(out_sum)
            print(output)

if __name__ == "__main__":
    input_nodes = 3
    hidden_layer_nodes = 4
    output_layer_nodes = 1
    hidden_activation = output_activation = "sigmoid"

    #instantiate show_model object
    neural_network = network_model(input_nodes,hidden_layer_nodes,
                                output_layer_nodes, hidden_activation,
                                output_activation)
    
    neural_network.view_pdf()
    neural_network.impliment_model()