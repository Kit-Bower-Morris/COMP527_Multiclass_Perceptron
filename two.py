import numpy as np
import random
import matplotlib.pyplot as plt
"""
Perceptron Class. 
"""
class Perceptron():
    """
    init function. 
    no_of_inputs = 4. - Number of varibles in data set. 
    threshold = 20. - Number of iterations.
    weights - start as all zeros [0,0,0,0]
    bias - starts as 0
    """
    def __init__(self, no_of_inputs=4, threshold=20):
        self.threshold = threshold
        self.weights = np.zeros(no_of_inputs)
        self.bias = 0

    """
    train fuction.
    Follow perceptron algorithm. 
    """
    def train(self, training_inputs, labels):
        errors = [] #List of amount of errors per epoch. 
        for _ in range(self.threshold):
            count = 0
            training_inputs, labels = Perceptron.shuffle(self, training_inputs, labels) #Shuffle order of input data.
            for inputs, label in zip(training_inputs, labels):
                a = np.dot(inputs, self.weights) + self.bias #Activation function.
                
                if((label*a) <= 0): #Checks the sign of the actication function. If it not the same as label, weight is adjusted. 
                    count += 1
                    self.weights += label * inputs #Update formula.
                    self.bias += label
            errors.append(count/len(training_inputs)) #Total number of errors for current epoch. 
        return self.bias, self.weights, errors

    """
    shuffle function. 
    Takes in data and corrisponding labels. 
    shuffles the order, keeping the data labels pair together. 
    """
    def shuffle(self, data, labels):
        z = list(zip(data, labels))
        random.shuffle(z) #Shuffle data, label pair. 
        data, labels = zip(*z)
        return data, labels

    """
    predict function.
    Tests the test data using the weights found in training.
    Returns percentage of correct predictions.
    """
    def predict(self, data, labels):
        a = np.dot(data, self.weights) + self.bias
        predictions = []
        result = []
        for i in a:
            if (i > 0):
                predictions.append(1)
            else:
                predictions.append(-1)  

        for i in range(len(labels)): #Compares predictions with answers.
            if labels[i] == predictions[i]:
                result.append(labels[i])
        
        return (len(result))/len(labels)
