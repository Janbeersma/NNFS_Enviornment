import sys
import numpy as np
import matplotlib as plt
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

#class voor het initializeseren van de dense layers in het neurale netwerk
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

#class voor het berkenen van de Rectified Linear Units outputs
class Activation_ReLU:
    #een forward functie die het doel heeft het berkenen van de ReLU die als input de daadwerkelijke inputs van het NN heeft
    #of de output van andere neuronen in de hidden layers
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

#class voor het bereken van de softmax activatie output
class Activation_Softmax:
    #een forward functie die inkomende data van andere neurons als input heeft als dit de classificatie neuron is
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

#normale loss class
class Loss:
    #berekend de data en ge-regulizeerde losses
    #gegeven model output en ground truth values
    def calculate(self, output, y):
        
        #berekend sample losses
        sample_losses = self.forward(output, y)

        #berkend gemiddelde loss
        data_loss = np.mean(sample_losses)

        #return loss
        return data_loss

#X, y is een spiral dataset met iedere class 100 samples verdeeld over drie verschillende classes
X, y = spiral_data(samples=100, classes=3)

#de eerste dense layer heeft 2 inputs en 3 neuronen
dense1 = Layer_Dense(2,3)

#activation1 is een ReLU activatiefunctie
activation1 = Activation_ReLU()

#de 2e dense layer heeft 3 inputs gezien de voorgaande layer 3 outputs had
#en deze worden weer output naar 3 neuronen
dense2 = Layer_Dense(3,3)

#activation2 is een Softmax activatiefunctie
activation2 = Activation_Softmax()

#De input van dense1 is dataset X
dense1.forward(X)

#de ReLU activatiefunctie wordt uitgevoerd op dense1 
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])