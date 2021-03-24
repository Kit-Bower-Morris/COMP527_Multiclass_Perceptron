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
Multiclass class.
This class converts the binary perceptron into a multiclass classifier. 
"""
class Multiclass(Perceptron):

    """
    init function.
    Initialises perceptron and enviroment class.
    """
    def __init__(self, file_name):
        super().__init__()
        self.env = Enviroment(file_name) #Calls Enviroment class.

    """
    training function. 
    self.env.data = sorted data
    self.recorded_bias = [0,0,0]
    self.recorded_errors = []
    self.recorded_weight = 3*4 zeros array

    Trains for each occurance of the multiclass and records the invidual weights, bias and errors. 
    """
    def training(self):
        self.env.data = self.env.read(self.env.file_name)
        self.recorded_bias = np.zeros((3))
        self.recorded_errors = []
        self.recorded_weight = np.zeros((3, 4))

        for i in range(3): #For each 1vsrest version.
            self.perceptron = Perceptron()
            data, labels = self.env.formatter(self.env.data)           
            labels = self.env.label(labels,i+1)
            bias, weight, errors = self.perceptron.train(data, labels)
            self.recorded_bias[i] += bias #Record Bias
            self.recorded_errors.append(errors) #Record Errors
            for k in range(len(weight)):
                self.recorded_weight[i][k] += weight[k] #Record Weights
        return self.recorded_errors

    """
    test function.
    Initialises enviroment with test data.
    Creates an array of all the activation values using the weights and bias found in the training. 
    """
    def test(self, file_name):
        self.env = Enviroment(file_name)
        self.env.data = self.env.read(self.env.file_name)
        self.confidence = np.zeros((len(self.env.data), 3))
        random.shuffle(self.env.data) #Shuffle data
        data, labels = self.env.formatter(self.env.data)
        self.labels = labels
        for i in range(3):
            class_weight = np.zeros((4))
            for j in range(4):
                class_weight[j] += self.recorded_weight[i][j]
            a = np.dot(data, class_weight) + self.recorded_bias[i]
            for j in range(len(a)):
                self.confidence[j][i] += a[j] #adds a to the confidence array. 
        return Multiclass.accuracy(self) #Call accuracy function.

    """
    accuracy function.
    Using the activation values it finds the maximum for each iteration.
    This then becomes the prediction for that data's classification.
    Then comparing these to the labels gives an accuracy as a percent. 
    """
    def accuracy(self):
        max_values = []
        result = []
        acc = []
        for i in range(len(self.confidence)):
            max_values.append((np.argmax(self.confidence[i])+1)) #Predict class using highest activation value. 
        
        for i in range(len(max_values)):
            if self.labels[i] == max_values[i]:
                result.append(self.labels[i])

        
        for i in range(1,4):
            acc.append((result.count(i))/10) #Counts how many of each class were correctly labelled.
        return acc



"""
Calls multiclass class and initialises the eviroment with the training data.
Plots errors in the training phase
Plots bar graph showing accuracy of the test. 
"""
def main():
    env = Multiclass("train.data")
    errors = env.training()
    acc = env.test("test.data")
    plt.plot(errors[0])
    plt.plot(errors[1])
    plt.plot(errors[2])
    plt.xlim(0,20)
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, 21, step=5))
    plt.ylabel('Errors')
    plt.xlabel('Epoch')
    plt.legend(["Class 1 vs Rest", "Class 2 vs Rest", "Class 3 vs Rest"], loc = 1, frameon = False)
    plt.show()


    objects = ('Class 1', 'Class 2', 'Class 3')
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, acc, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.ylabel('Correct')

    plt.show()

if __name__ == "__main__":
    main()
