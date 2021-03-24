import numpy as np
import random

class Perceptron():

    def __init__(self, no_of_inputs=4, threshold=20):
        self.threshold = threshold
        self.weights = np.zeros(no_of_inputs)
        self.bias = 0
           
    def predict(self, inputs):
        a = np.dot(inputs, self.weights) + self.bias
        predictions = []
        for i in a:
            if (i > 0):
                predictions.append(1)
            else:
                predictions.append(-1)  
        return predictions

    def classifier(self, inputs):
        a = np.dot(inputs, self.weights) + self.bias
        return a

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                a = np.dot(inputs, self.weights) + self.bias
                
                if((label*a) <= 0):
                    self.weights += label * inputs
                    self.bias += label
        return self.bias, self.weights

class Enviroment():
    
    def __init__(self, file_name):
        self.data = Enviroment.read(self, file_name)
        self.perceptron = Perceptron()

    def sort(self, data):
        start = []
        for i in data:
            start.append(i)
        return start

    def read(self, x):
        f = open(x, "r")
        data = f.read().splitlines()
        f.close()
        return Enviroment.sort(self, data)


    def go(self, first, second):
        start = Enviroment.compare(self, first, second)
        data, labels = Enviroment.initial(self,start)
        labels = Enviroment.label(self,labels,first)
        bias, weight = self.perceptron.train(data, labels)
        return bias, weight

    def compare(self, first, second):
        start = []
        for i in self.data:
            if((int(i[22])) == first or (int(i[22]) == second)):
                start.append(i)
        random.shuffle(start)
        return start

    def initial(self, start):
        data = []
        labels = []
        for i in start: 
            train = []
            train.append(float(i[:3]))
            train.append(float(i[4:7]))
            train.append(float(i[8:11]))
            train.append(float(i[12:15]))
            data.append(np.array(train))
            labels.append((int(i[22])))

        return data, labels
        
    def label(self, labels, first):

        for i in range(len(labels)):            
            if labels[i] == first:
                labels[i] = 1
            else:
                labels[i] = -1           

        return labels

    

class Multiclass(Perceptron):

    def __init__(self, file_name, classes):
        super().__init__()
        self.env = Enviroment(file_name)
        self.classes = classes
        test = self.env.data
        #random.shuffle(test)

        self.recorded_bias = np.zeros((classes))
        self.recorded_weight = np.zeros((3, 4))

        
        for i in range(self.classes):
            data, labels = self.env.initial(test)
            labels = self.env.label(labels, i+1)
            bias, weight = Perceptron.train(self, data, labels)
            self.recorded_bias[i] += bias
            for k in range(len(weight)):
                self.recorded_weight[i][k] += weight[k]

        

    def test(self, file_name):
        self.env = Enviroment(file_name)
        test = self.env.data
        random.shuffle(test)
        data, labels = self.env.initial(test)
        x = np.zeros((len(test), self.classes))
        for i in range(self.classes):
            class_weight = np.zeros((4))
            for j in range(4):
                class_weight[j] += self.recorded_weight[i][j]
            #print(class_weight)
            a = np.dot(data, class_weight) + self.recorded_bias[i]
            for j in range(len(a)):
                x[j][i] += a[j]
        
        result = []
        print(labels)
        for i in range(len(x)):
            result.append(np.argmax(x[i]))
        
        for i in range(len(labels)):
            if labels[i] == result[i]:
                labels[i] = 1
            else:
                labels[i] = 0
        
        count = 0
        for i in range(len(labels)):
            if labels[i] == 1:
                count += 1


        return result
        


            
        
"""        

for i in range(len(x)):
    result.append(np.argmax(x[i]))

final = []
one = 0
two = 0
three = 0
for i in result:
    if i == 0:
        one += 1
    if i == 1:
        two += 1
    if i == 2:
        three += 1
final.append((one, two, three))
#print(final)
print(memory)
"""

 
            



    









def main():
    #env = Enviroment("train.data")
    #print(env.go(1,2))
    #print(env.go(1,3))
    #print(env.go(2,3))

    hi = Multiclass("train.data",3)
    print(hi.test("test.data"))
    

    """
    env = Enviroment("test.data")
    data, label = env.initial(1,2)
    print(data)
    print(label)
    print(perceptron.predict(data))
    """

    
    """
    train = read("train.data")
    env = Enviroment(train)
    perceptron = Perceptron(4)
    data, labels = env.initial(2,3)
    print(perceptron.train(data, labels))

    test = read("test.data")
    env = Enviroment(test)
    data, label = env.initial(2,3)
    print(label)
    print(perceptron.predict(data))

    train = read("train.data")
    env = Enviroment(train)
    perceptron = Perceptron(4)
    data, labels = env.initial(1,3)
    print(perceptron.train(data, labels))

    test = read("test.data")
    env = Enviroment(test)
    data, label = env.initial(1,3)
    print(label)
    print(perceptron.predict(data))
    """





if __name__ == "__main__":
    main()





     
        