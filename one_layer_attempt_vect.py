import numpy as np 
import matplotlib.pyplot as plt 
import math as m 
import random  
import pandas as pd 


pi = m.pi
learning_rate = 0.0075
no_inputs = 1
no_outputs = 1

def Sine_data():

    """

    Function that generates data for toy problem and stores it in CSV file

    """


    data = np.linspace(-2*pi,2*pi,1000)

    data = data.reshape(1,1000)

    out = np.sin(data)

    

    return data,out

def tanhx(data):

    """
    Function that applies the tanhx function to all elements of a numpy array

    Input: Array data
    Output " Array with tanhx values

    """

    output = np.tanh(data)

    return output

def tanhx_der(data):
    """

    Function that returns the derivative of the tanhx function 
    Input : Data point 
    Output : Derivative of tanhx at point
    """

    
    return 1 - tanhx(data)**2

def feedforward_output(input_val,weights,bias):

    output_layer = tanhx(np.dot(weights,input_val) + bias)

    return output_layer

def backprop(activations,weights,biases,exact_outputs):

    
    dz_out = activations[-1] - exact_outputs
    dw_out = 0.001*np.dot(dz_out,activations[-2].T)
    db_out = 0.001*np.sum(dz_out,axis=1,keepdims=True)
    dz_hidden = np.multiply(np.dot(weights[-1].T,dz_out),tanhx_der(activations[-2]))
    dw_hidden = 0.001*np.dot(dz_hidden,activations[0].T)
    db_hidden = 0.001*np.sum(dz_hidden,axis=1,keepdims=True)

   

    

    weights[-1] = weights[-1] - learning_rate*dw_out
    weights[-2] = weights[-2] - learning_rate*dw_hidden
    biases[-1] = biases[-1] - learning_rate*db_out
    biases[-2] = biases[-2] - learning_rate*db_hidden

    return weights,biases


def Train_network(input_data,output_data):

    """
    A function that trains the network
    """

    hidden_size = 100


    weights_0_1 = np.random.randn(hidden_size,no_inputs)*0.1
    weights_0_2 = np.random.randn(no_outputs,hidden_size)*0.1
    bias_1 = np.zeros(shape = (hidden_size,1))
    bias_2 = np.zeros(shape = (no_outputs,1))
    weights = np.array([weights_0_1,weights_0_2])
    biases = np.array([bias_1,bias_2])
    
    
    

    for epoch in range(5000):

        hidden_layer = feedforward_output(input_data,weights[0],biases[0])
        output_layer = feedforward_output(hidden_layer,weights[1],biases[1])
        activations = np.array([input_data,hidden_layer,output_layer])

        

        weights,biases =  backprop(activations,weights,biases,output_data)



        output = "Epoch %r Error %r"%(epoch,0.001*np.mean((output_layer-output_data)**2)/2)
        print(output)

    return weights,biases


if __name__ == "__main__":

    input_data,output_data = Sine_data()

    weights,biases= Train_network(input_data,output_data)

    #all_in = np.array([])

   
    #input_layer_normalized = (2*input_layer)/4*pi

    

    
    hidden_layer = tanhx(np.dot(weights[0],input_data) + biases[0])
    output_layer = tanhx(np.dot(weights[1],hidden_layer) + biases[1])

            

    plt.plot(input_data[0],output_data[0],'r')
    plt.plot(input_data[0],output_layer[0],'g')
    plt.show()