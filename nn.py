import numpy as np 

class neural_network:
    def __init__(self , shape , learning_rate = 0.001,act_fn = 'relu'):
        self.alpha             = learning_rate
        self.weights           = []
        
        self.shape             = shape
        
        self.layers            = len(self.shape)
        
        self.neurons           = [] 
        self.activations       = []
        self.gradients         = []
        self.grad_sum          = []
        self.hypothesis_errors = []
        self.errors = []

        print("\n##############################################################- #   ###-################################################################")
        print("\n\nNETWORK STRUCTURE:\n")
        for layer in range(self.layers - 1):
            print("layer - %3d" %(layer) , "  ---- %4d" %(self.shape[layer]),"-", act_fn , " neurons")
        print("layer - %3d" %(self.layers - 1) , "  ---- %4d" %(self.shape[-1]) ,"- sigmoid neurons\n\n\n")
        print("##############################################################- #   ###-################################################################\n\n")


        if act_fn == 'sigmoid':
            self.activation_function = self.sigmoid_function
            self.derivative_function = self.sigmoid_derivative
        elif act_fn == 'relu':
            self.activation_function = self.relu_function
            self.derivative_function = self.relu_derivative

        self.initialize()


    #function to initialize weights,biases,neurons
    def initialize(self):
        
        #a extra bias neuron with value 1 is added to each layer
        #neurons and weight matrix initialization with bias
        for i in range(self.layers - 1):
            self.weights.append(np.random.random((self.shape[i] + 1 , self.shape[i + 1])))
            self.grad_sum.append(       np.zeros((self.shape[i] + 1 , self.shape[i + 1])))
            self.neurons.append(          np.ones(self.shape[i] + 1))

        self.neurons.append(np.ones(self.shape[-1] + 1))

        self.neurons = (np.array(self.neurons))
        self.weights = (np.array(self.weights))


    def feed_forward(self,inputs):
        #update neurons matrix input layer
        self.neurons[0][1:] = inputs
        self.activations.append(np.array(self.neurons[0]))
    
        #activation = neuron*weight + bias
        for layer in range(self.layers  - 2):
            self.neurons[layer + 1][1:] = np.matmul((self.neurons[layer]) , (self.weights[layer]))
            self.activations.append(np.array(self.neurons[layer + 1]))
            #passing through activation function
            self.neurons[layer + 1][1:] =  np.array(list(map(self.activation_function , self.neurons[layer + 1][1:])))

        #final layer sigmoid always
        self.neurons[-1][1:] = np.matmul((self.neurons[-2]) , (self.weights[-1]))
        self.activations.append(np.array(self.neurons[-1]))
        self.neurons[-1][1:] =  np.array(list(map(self.sigmoid_function , self.neurons[-1][1:])))

        #return all layer activations,outputs and final hypothesis
        return [self.activations,self.neurons,self.neurons[-1][1:]] 

    def backpropagate(self,delta):
        self.errors = []
        self.errors.append(delta)
       
        for layer in range(self.layers - 2 , 0 , -1):
            tempdelta = np.matmul(np.array(self.weights[layer][1:]) , self.errors[self.layers - 2 - layer])
            self.errors.append(tempdelta * np.array(list(map(self.derivative_function , self.neurons[layer][1:]))))

        self.calc_gradients()
        # self.accumulate_gradients()
        # self.sgd()

        return self.gradients


    def calc_gradients(self):
        self.gradients = []
        for layer in range(self.layers - 1):
            self.gradients.append(np.matmul(np.array([self.neurons[layer]]).T , np.array([self.errors[self.layers - 2 - layer]]))) 
  

    def sigmoid_function(self,z):
        return (1 / (1 + np.exp(-z)))


    def sigmoid_derivative(self,fn):
        return fn * (1 - fn)

    def relu_function(self,z):
        return np.max([0.0 , z])

    def relu_derivative(self,fn):
        return 1.0 if fn > 0 else 0.0

    def MSELoss(self,outputs):
        temp1 = ((np.array(self.neurons[-1][1:]) - np.array(outputs)))
        self.hypothesis_errors = temp1 ** 2
        return [temp1,(np.sum(temp1 ** 2))/(2*self.shape[-1])]

    def BCELoss(self,outputs):
        outputs = np.array(outputs)
        delta   = (self.neurons[-1][1:] - outputs)
        log     =  outputs * np.log(self.neurons[-1][1:])
        onelog  = (1 - outputs) * np.log(1 - self.neurons[-1][1:])
        error   = -(log + onelog)
        loss    = np.sum(error / self.shape[-1])
        return loss , delta
 

    def sgd(self):
        for layer in range(self.layers-1):
            self.weights[layer] = (self.weights[layer] - (self.alpha * np.array(self.gradients[layer])))

    def accumulate_gradients(self):
        for layer in range(self.layers-1):
            self.grad_sum[layer] = (self.grad_sum[layer] + np.array(self.gradients[layer]))

    # def mbgd(self , batch_size):
    #     for layer in range(self.layers-1):
    #         self.weights[layer] = (self.weights[layer] - (self.alpha * np.array(self.grad_sum[layer]) / batch_size))
    #         self.grad_sum[layer] = (np.zeros((self.shape[layer] + 1 , self.shape[layer + 1])))








if __name__ == '__main__':

    import pandas as pd 


    f  = pd.read_csv("f://iris_training.csv")
    f  = np.array(f)

    ip = f[:,:4]
    op = f[:,4]

    index = np.arange(ip.shape[0])
    np.random.shuffle(index)

    ip = ip[index].astype(np.float32)
    op = op[index].astype(np.int64)

    target = np.zeros((len(ip) , 3))
    for i in range(len(target)):
        target[i,op[i]] = 1.0




    #network declaration
    net = neural_network(shape = [4,10,20,8,3] , learning_rate = 0.001 , act_fn = 'relu')


    #training
    for i in range(100):
        j = 0
        for iput , oput in zip(ip , target):
            j += 1
            out = net.feed_forward(iput)
            loss , delta = net.BCELoss(oput)
            g = net.backpropagate(delta)
            net.sgd()
            print('[Epoch] : [', 100 ,"/", i ,"]\tloss : " , loss)

            # print(i ,":" , j ," : loss:",loss)

    #testing
    print("\n\nTesting\n\n")
    for i in range(len(target)):
        otpt = net.feed_forward(ip[i])[-1]
        print("output of",i, "th input", np.argmax(otpt))