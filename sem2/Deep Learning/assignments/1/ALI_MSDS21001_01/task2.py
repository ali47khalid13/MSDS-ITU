# Include libraries which may use in implementation
import numpy as np
import random
import sklearn.datasets as ds
import matplotlib.pyplot as plt
np.random.seed(7)

# Create a Neural_Network class
class Neural_Network(object):        
    def __init__(self,inputSize = 2,hiddenLayer = 3 ,outputSize = 1, activation = 'sigmoid'):        
        # size of layers
        self.inputSize = inputSize
        self.outputSize = outputSize 
        self.hiddenLayer = hiddenLayer
   
        if activation== 'sigmoid':
            self.activation = self.sigmoid;
            self.activation_d = self.sigmoid_derivative;
        elif activation== 'tanh':
            self.activation = self.tanh;
            self.activation_d = self.tanh_derivative;
        elif activation== 'relu':
            self.activation = self.relu;
            self.activation_d = self.relu_derivative;
        else:
            self.activation = self.sigmoid;
            self.activation_d = self.sigmoid_derivative;
        #weights
        self.W1 = np.random.rand(self.inputSize +1, self.hiddenLayer) # randomly initialize W1 using random function of numpy
        # size of the wieght will be (inputSize +1, hiddenlayer) that +1 is for bias    
        self.W2 = np.random.rand(self.hiddenLayer +1, self.outputSize) # randomly initialize W2 using random function of numpy
        # size of the wieght will be (hiddenlayer +1, outputSize) that +1 is for bias    
        
    def feedforward(self, X):
        #forward propagation through our network
        # dot product of X (input) and set of weights
        # apply activation function (i.e. whatever function was passed in initialization)    

        hiddenlayer_output = self.activation(np.dot(X,self.W1))
        hiddenlayer_output = np.c_[ hiddenlayer_output, np.ones(len(hiddenlayer_output)) ] 
        outputlayer_output = self.sigmoid(np.dot(hiddenlayer_output,self.W2))
        return hiddenlayer_output, outputlayer_output # return your answer with as a final output of the network

    def sigmoid(self, s):
        # activation function
        z = 1/(1 + np.exp(-s))
        return z # apply sigmoid function on s and return it's value

    def sigmoid_derivative(self, s):
        #derivative of sigmoid
        ds = s*(1-s)
        return ds # apply derivative of sigmoid on s and return it's value 
    
    def tanh(self, s):
        # activation function
        t= (np.exp(s)-np.exp(-s))/(np.exp(s)+np.exp(-s))
        return t # apply tanh function on s and return it's value

    def tanh_derivative(self, s):
        #derivative of tanh
        dt= 1-self.tanh(s)**2
        return dt # apply derivative of tanh on s and return it's value
    
    def relu(self, s):
        # activation function
        r =  s * (s > 0)
        return r # apply relu function on s and return it's value

    def relu_derivative(self, s):
        #derivative of relu
        dr =  1. * (s > 0)
        return dr # apply derivative of relu on s and return it's value

    def backwardpropagate(self,X, Y, y_pred, lr,hiddenlayer_output):
        # backward propagate through the network
        # compute error in output which is loss compute cross entropy loss function
        y_pred = np.squeeze(y_pred)
        cross_entropy_d = (-(Y/y_pred)) + ((1-Y)/(1-y_pred))
        dL_dypred = np.multiply(self.sigmoid_derivative(y_pred),cross_entropy_d)
        dL_dypred =  dL_dypred.reshape(len(dL_dypred),1)

        # applying derivative of that applied activation function to the error
        update_w2 = np.dot(dL_dypred.transpose(),hiddenlayer_output).transpose()
        update_w1 = np.dot(X.transpose(),np.multiply(self.activation_d(hiddenlayer_output),(np.dot(dL_dypred,self.W2.transpose())))[:,0:-1])

        # adjust set of weights
        self.W1 -= lr*(update_w1/len(Y))
        self.W2 -= lr*(update_w2/len(Y))
    
    def crossentropy(self, Y, Y_pred):
        # compute error based on crossentropy loss 
        loss =  -np.dot(Y,np.log(Y_pred)) - np.dot((1-Y),np.log(1-Y_pred))
        return loss/len(Y) #error

    def train(self, trainX, trainY,epochs = 10000, learningRate = 0.1, plot_err = False ,validationX = None, validationY = None):
        # feed forward trainX and trainY and recivce predicted value
        train_loss_history = []
        val_loss_history = []
        train_acc_history =[]
        val_acc_history =[]
       
        for epoch in range(epochs):

            #feedforward from network
            hiddenlayer_output, y_pred = self.feedforward(trainX)

            # backpropagation with trainX, trainY, predicted value and learning rate.
            self.backwardpropagate(trainX,trainY, y_pred, learningRate, hiddenlayer_output)

            #compute loss and accracy
            train_loss = self.crossentropy(trainY,np.squeeze(y_pred))
            y_pred = self.predict(trainX)
            accuracy_train = np.sum(y_pred == trainY) /len(trainY)
            train_loss_history.append(train_loss)
            train_acc_history.append(accuracy_train)

            # if validationX and validationY are not null than show validation accuracy and error of the model by printing values.
            if (validationY.all()!= None):
                _ , y_pred = self.feedforward(validationX)
                val_loss = self.crossentropy(validationY,np.squeeze(y_pred))
                y_pred = self.predict(validationX)
                accuracy_val = np.sum(y_pred == validationY) /len(validationY)
                val_loss_history.append(val_loss)
                val_acc_history.append(accuracy_val)
                print("Epoch = %3d   Train Loss = %3.3f  Val Loss=%3.3f" %(epoch, train_loss_history[epoch],val_loss_history[epoch]))
            else:
                print("Epoch = %3d   Train Loss = %3.3f" %(epoch, train_loss_history[epoch]))

        # plot error of the model if plot_err is true
        if (plot_err==True):
            plot_curves(train_loss_history, val_loss_history, 'Training VS Validation loss', 'Epoch', 'Loss')
            plot_curves(train_acc_history, val_acc_history, 'Training VS Validation Accuracy', 'Epoch', 'Accuracy')
            #plot_losses(val_loss_history, 'Validation loss')


    def predict(self, testX):
        # predict 1 if output is grater than 0.5 otherwise predict 0
        _, y_pred = self.feedforward(testX)
        y_pred = np.squeeze(y_pred)
        y_pred = [1 if y > 0.5 else 0 for y in y_pred]
        return y_pred

    
    def accuracy(self, testX, testY, trainX, trainY):
        # predict the value of trainX and testX
        y_pred_test= self.predict(testX)
        y_pred_train= self.predict(trainX)
        # compute accuracy, print it and show in the form of picture
        accuracy_test = np.sum(y_pred_test == testY) /len(testY)
        accuracy_train = np.sum(y_pred_train == trainY) /len(trainY)
        print('Accuracy of model on test data:', accuracy_test)
        print('Accuracy of model on train data:', accuracy_train)
        label = ['test', 'train']
        accuracy = [accuracy_test, accuracy_train]
        plt.bar(label, accuracy, width=0.8, align='center')
        plt.xlabel("Sample", fontsize=15)
        plt.ylabel("Accuracy", fontsize=15)
        plt.title('Train vs Test Accuracy')
        plt.show()
        return accuracy# return accuracy    
        
    def saveModel(self,name):
        # save your trained model, it is your interpretation how, which and what data you store
        # which you will use later for prediction    
        file_name = name+'.npy'
        with open(file_name, 'wb') as f:
            np.save(f, self.W1)
            np.save(f, self.W2)

    def loadModel(self,name):
        # load your trained model, load exactly how you stored it.
        file_name = name+'.npy'
        with open(file_name, 'rb') as f:
            self.W1= np.load(f)
            self.W2 = np.load(f)


def train_val_test_split(data, label, train_per, val_per):
    # This function is used to split the data into test, train and validation
    samples = len(data)
    train_samples = np.ceil(samples*train_per).astype('int32')
    val_samples = np.ceil(samples*val_per).astype('int32')
    indices = np.random.permutation(samples)
    train_idx, val_idx, test_idx = indices[:train_samples], indices[train_samples:train_samples+val_samples] , indices[train_samples+val_samples:]
    trainx, valx, testx = data[train_idx], data[val_idx], data[test_idx]
    trainy, valy, testy = label[train_idx], label[val_idx], label[test_idx]
    return trainx, trainy, valx, valy, testx, testy


def plot_curves(train_loss, val_loss, title, X_label, Y_label):
    # This function is used to plot the loss
    plt.plot(train_loss, linewidth=2, label='train')
    plt.plot(val_loss, linewidth=2, label='validation')
    plt.xlabel(X_label, fontsize=15)
    plt.ylabel(Y_label, fontsize=15)
    plt.legend()
    plt.title(title)
    plt.show()

def main():   

    # Creata Dataset 
    data, label = ds.make_circles(n_samples=1000, factor=.4, noise=0.05)
    # Append coloumns of 1 in data for bias
    data = np.c_[ data, np.ones(len(data)) ] 

    # Visualize the Data
    # reds = label == 0
    # blues = label == 1
    # plt.scatter(data[reds, 0], data[reds, 1], c="red", s=20, edgecolor='k')
    # plt.scatter(data[blues, 0], data[blues, 1], c="blue", s=20, edgecolor='k')
    # plt.show()

    # Distribute this data into three parts i.e. training, validation and testing
    trainX , trainY, validX, validY, testX, testY= train_val_test_split(data, label,0.70,0.15)
   
    # Create neural Network 
    model = Neural_Network(2,3,activation = 'sigmoid')

    # try different combinations of epochs and learning rate
    model.train(trainX, trainY, epochs = 1000, learningRate = 0.1, validationX = validX, validationY = validY, plot_err = True)
    #print(model.W2[0:10])
    #save the best model which you have trained, 
    model.saveModel('bestmodel_task2')
  

    # load model which will be provided by you
    # create class object
    mm = Neural_Network(2,3,activation = 'sigmoid')
    mm.loadModel('bestmodel_task2')
    # check accuracy of that model
    mm.accuracy(testX,testY,trainX,trainY)


if __name__ == '__main__':
    main()