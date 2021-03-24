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



"""
Enviroment Class. 
Reads and formats input data. 
"""
class Enviroment():
    
    def __init__(self, file_name):
        self.file_name = file_name

    """
    read function. 
    Opens file, converts each line to a string and closes file.
    Calls the sort fuction. 
    Returns sorted data. 
    """  
    def read(self, file_name):
        f = open(file_name, "r")
        read_data = f.read().splitlines()
        f.close()
        return Enviroment.sort(self, read_data) #Calls sort function. 

    """
    sort function. 
    converts the read data into a list. 
    """   
    def sort(self, data):
        read_data = []
        for i in data:
            read_data.append(i)
        return read_data

    """
    classifier function.
    Takes the two classes that will be compared and returns just the data that includes these two classes. 
    """
    def classifier(self, first, second):
        classified_data = []
        for i in self.data:
            if((int(i[22])) == first or (int(i[22]) == second)): #Removes third class.
                classified_data.append(i) 
        return classified_data    
    
    """
    formatter function.
    Formats data into the appropriate format. [float,float,float,float]
    Seperates the label from data and records this at the same index. 
    """
    def formatter(self, data):    
        formatted_data = []
        labels = []
        for i in range(len(data)): 
            train = []
            train.append(float(data[i][:3]))
            train.append(float(data[i][4:7]))
            train.append(float(data[i][8:11]))
            train.append(float(data[i][12:15]))
            formatted_data.append(np.array(train))
            labels.append((int(data[i][22]))) #Label of data

        return formatted_data, labels

    """
    label function. 
    Converts labels into binary format.
    First = label which will be vs the rest.
    """
    def label(self, labels, first):
        for i in range(len(labels)):            
            if labels[i] == first:
                labels[i] = 1 #postive class
            else:
                labels[i] = -1 #negative class        
        return labels

    """
    initialiser function.
    Calls and runs all functions for the training of the perceptron.
    inputs are the two classes that will be compared.
    """
    def initialiser(self, first, second):
        self.data = Enviroment.read(self, self.file_name)
        self.perceptron = Perceptron() #Initialises perceptron.
        classified_data = Enviroment.classifier(self,first,second)
        data, labels = Enviroment.formatter(self,classified_data)
        labels = Enviroment.label(self,labels,first)
        bias, weight, errors = self.perceptron.train(data, labels)
        return bias, weight, errors

    """
    test function.
    Calls and runs all functions for the test of the perceptron.
    inputs are the two classes that will be compared.
    """
    def test(self, file_name, first, second):
        self.data = Enviroment.read(self,file_name)
        classified_data = Enviroment.classifier(self,first,second)
        random.shuffle(classified_data)
        data, labels = Enviroment.formatter(self,classified_data)
        labels = Enviroment.label(self,labels,first)
        return self.perceptron.predict(data, labels)



"""
Trains and tests for each different class comparison. 
Plots graphs for these.
"""
def main():
    env = Enviroment("train.data")
    acc = []
    test = "test.data"
    _, _, errors1 = env.initialiser(1,2)
    acc.append(env.test(test,1,2))
    _, _, errors2 = env.initialiser(2,3)
    acc.append(env.test(test,2,3))
    _, _, errors3 = env.initialiser(1,3)
    acc.append(env.test(test,1,3))
    plt.plot(errors1)
    plt.plot(errors2)
    plt.plot(errors3)
    plt.xlim(0,20)
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, 21, step=5))
    plt.ylabel('Errors')
    plt.xlabel('Epoch')
    plt.legend(["(1,2)", "(2,3)", "(1,3)"], loc = 1, frameon = False)
    plt.show()
    

    objects = ('Class 1 vs Class 2', 'Class 2 vs Class 3', 'Class 1 vs Class 3')
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, acc, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.ylabel('Percent Correct')

    plt.show()



if __name__ == "__main__":
    main()
