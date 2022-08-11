# Include libraries which may use in implementation
import numpy as np
import glob
import random
import sklearn.datasets as ds
from matplotlib import image as img
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import pandas as pd  
np.random.seed(211)

# Create a Layer class
class layer():
    def __init__(self,insize,outsize):
        # create class attributes to store weights, bias, activation output, local gradient, derivative of weights and bias
        self.W = np.random.rand(insize, outsize)
        self.b = np.random.rand(outsize)
        self.a = np.array([])
        self.local_grad = np.array([])
        self.dw = np.zeros((insize, outsize))
        self.db = np.zeros(outsize)

# Create a Network class
class init_network(object):  
    def __init__(self, no_of_layers , input_dim = 784, neurons_per_layer = [128, 64 , 6]):   

        # Dynamically Initialize number of layers with specifed number of neurons
        self.layer = [None]*(no_of_layers+1)
        self.input_dim = input_dim

        for i in range(len(neurons_per_layer)):
            if i == 0:
                self.layer[i] = layer(input_dim,neurons_per_layer[i])
            else:
                self.layer[i] = layer(neurons_per_layer[i-1],neurons_per_layer[i])
        for i in range(len(neurons_per_layer)):
            print(self.layer[i].W.shape)
    
def feedforward(net, data):

    # perform forward pass and also store activation and local gradient for each layer

    for i in range(len(net.layer)):
        if i == 0:
            net.layer[i].a = sigmoid(np.dot(data,net.layer[i].W) + net.layer[i].b)
            net.layer[i].local_grad = sigmoid_derivative(net.layer[i].a)
        elif i<len(net.layer)-1:
            net.layer[i].a = sigmoid(np.dot(net.layer[i-1].a,net.layer[i].W) + net.layer[i].b)
            net.layer[i].local_grad = sigmoid_derivative(net.layer[i].a)
        else:
            net.layer[i].a = np.dot(net.layer[i-1].a,net.layer[i].W) + net.layer[i].b
            net.layer[i].local_grad = np.ones(net.layer[i].a.shape)

    return net 

def softmax(x):  
    max = np.max(x,axis=1,keepdims=True) #returns max of each row and keeps same dims
    e_x = np.exp(x - max) #subtracts each row with its max value
    sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / sum 
    return f_x

def sigmoid(s):
    # activation function
    z = 1/(1 + np.exp(-s))
    return z # apply sigmoid function on s and return it's value

def sigmoid_derivative(s):
    #derivative of sigmoid
    ds = s*(1-s)
    return ds # apply derivative of sigmoid on s and return it's value 

def backPropagation(net,data,labels):

    loss = crossentropy(labels,net.layer[-1].a)

    prev_L_grad = delta_cross_entropy(labels,net.layer[-1].a)

    for i in range(len(net.layer)-1,-1,-1):
        if i == len(net.layer)-1:
            grad = np.multiply(net.layer[i].local_grad,prev_L_grad)
            net.layer[i].dw = np.dot(net.layer[i-1].a.T,grad)/len(data)
            net.layer[i].db = np.sum(grad,axis=0)/len(data)
            prev_L_grad = grad
        elif i > 0:
            grad = np.multiply(net.layer[i].local_grad,np.dot(prev_L_grad,net.layer[i+1].W.T))
            net.layer[i].dw = np.dot(net.layer[i-1].a.T,grad)/len(data)
            net.layer[i].db = np.sum(grad,axis=0)/len(data)
            prev_L_grad = grad
        elif i ==0:
            grad = np.multiply(net.layer[i].local_grad,np.dot(prev_L_grad,net.layer[i+1].W.T))
            net.layer[i].dw = np.dot(data.T,grad)/len(data)
            net.layer[i].db = np.sum(grad,axis=0)/len(data)
            prev_L_grad = grad
    return net, loss

def crossentropy(Y, Y_pred):
    m = Y.shape[0]
    p = softmax(Y_pred)
    log_likelihood = -np.log(p[range(m),Y])
    loss = np.sum(log_likelihood) / m
    return loss

def delta_cross_entropy(Y,Y_pred):
    m = Y.shape[0]
    grad = softmax(Y_pred)
    grad[range(m),Y] -= 1
    grad = grad/m
    return grad

def sgd(net, lr):
    for i in range(len(net.layer)):
        net.layer[i].W -= lr*net.layer[i].dw
        net.layer[i].b -= lr*net.layer[i].db
    return net

def tsne_plot(net,X, y , i, title):
    net = feedforward(net,X)
    tsne = TSNE(n_components=2, verbose=0, random_state=123)
    z = tsne.fit_transform(net.layer[i].a)
    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    plt.figure(figsize=(16,10))
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),data=df,palette=sns.color_palette("hls", 10))
    plt.title(title)
    plt.show()
    #.set(title="MNIST data T-SNE projection")


def test(net, test_set_x, test_set_y):
    net = feedforward(net,test_set_x)
    y_pred = np.argmax(net.layer[-1].a, axis=1)
    accuracy = np.sum(y_pred == test_set_y) /len(test_set_y)
    loss = crossentropy(test_set_y,net.layer[-1].a)
    return loss, accuracy

def predict(net, testX):
    # predict the value of testX
    net = feedforward(net,testX)
    #y_pred = softmax(net.layer[-1].a)
    y_pred = np.argmax(net.layer[-1].a, axis=1)
    return y_pred
   

def accuracy(net, testX, testY, trainX, trainY):
        # predict the value of trainX and testX
        y_pred_test= predict(net,testX)
        y_pred_train= predict(net,trainX)
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
    
def saveModel(net,name):   
    file_name = name+'.npy'
    with open(file_name, 'wb') as f:
        np.save(f, net.layer)

def loadModel(net,name):
    # load your trained model, load exactly how you stored it.
    file_name = name+'.npy'
    with open(file_name, 'rb') as f:
        net.layer = np.load(f,allow_pickle=True)

def saveArray(array,name):   
    file_name = name+'.npy'
    with open(file_name, 'wb') as f:
        np.save(f,array)

def loadArray(name):   
    array = []
    file_name = name+'.npy'
    with open(file_name, 'wb') as f:
        array = np.load(f)
    return array


def train_val_split(data, label, train_per):
    samples = len(data)
    train_samples = np.ceil(samples*train_per).astype('int32')
    indices = np.random.permutation(samples)
    train_idx, val_idx = indices[:train_samples], indices[train_samples:]
    trainx, valx = data[train_idx], data[val_idx]
    trainy, valy = label[train_idx], label[val_idx]
    return trainx, trainy, valx, valy

def plot_curves(train_loss, val_loss, title, X_label, Y_label):
    # This function is used to plot the loss
    plt.plot(train_loss, linewidth=2, label='train')
    plt.plot(val_loss, linewidth=2, label='validation')
    plt.xlabel(X_label, fontsize=15)
    plt.ylabel(Y_label, fontsize=15)
    plt.legend()
    plt.title(title)
    plt.show()

def train_sgd(net, train_set_x, train_set_y, val_set_x, val_set_y, learning_rate,  batch_size, training_epochs):
    # feed forward trainX and trainY and recivce predicted value
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    batches = np.ceil(len(train_set_y)/batch_size).astype('int32')
    batch_indices = np.random.permutation(len(train_set_y))
    for epoch in range(training_epochs):
        loss = 0 
        acc = 0
        for batch in range(batches):
            if batch == batches-1:
                X = train_set_x[batch_indices[batch_size*batch:-1]]
                Y = train_set_y[batch_indices[batch_size*batch:-1]]
            else:
                X = train_set_x[batch_indices[batch_size*batch:batch_size*(batch+1)]]
                Y = train_set_y[batch_indices[batch_size*batch:batch_size*(batch+1)]]
            net = feedforward(net,X)
            y_pred = np.argmax(net.layer[-1].a, axis=1)
            accuracy_train = np.sum(y_pred == Y) /len(Y)
            # backpropagation with trainX, trainY, predicted value and learning rate.
            net, cost = backPropagation(net,X,Y)
            loss += cost
            acc += accuracy_train
            # upgrading 
            net = sgd(net,learning_rate)

        train_loss_history.append(loss/batches)
        train_acc_history.append(acc/batches)
        val_loss, val_acc = test(net, val_set_x, val_set_y)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        print("Epoch = %3d   Train Loss = %3.3f  Val Loss=%3.3f" %(epoch, train_loss_history[epoch],val_loss_history[epoch]))
    return net, train_loss_history, val_loss_history, train_acc_history, val_acc_history

def meansubtraction(trainX, testX):
    data = np.concatenate((trainX, testX), axis=0)
    mean = np.mean(data, axis = 0)
    trainX = trainX-mean
    testX = testX-mean
    return trainX, testX

def loadDataset(path):
    print('Loading Dataset...')
    train_x, train_y, test_x, test_y = [], [], [], []
    for i in range(10):
        for filename in glob.glob(path + '\\train\\' + str(i)+'\\*.png'):
            im=img.imread(filename)
            train_x.append(im)
            train_y.append(i)
    for i in range(10):
        for filename in glob.glob(path + '\\test\\' + str(i)+'\\*.png'):
            im=img.imread(filename)
            test_x.append(im)
            test_y.append(i)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    print('  Dataset loaded')

    # flatten the vectors
    train_x = train_x.reshape(len(train_x),-1)
    test_x = test_x.reshape(len(test_x),-1)

    print('Subtracting Mean ...')
    train_x, test_x  = meansubtraction(train_x, test_x)
    print('  Subtracting Mean Done')

    return train_x, train_y, test_x, test_y

def confusion_matrix_plot(net,X,y):
    #classes = [0,1,2,3,4,5,6,7,8,9]
    predictions = predict(net,X)
    cm = confusion_matrix(y, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

def main():   

    # load dataset 
    trainX ,trainY, testX, testY = loadDataset('Task3_MNIST_Data')
    #Task3_MNIST_Data
    trainX ,trainY, valX, valY =  train_val_split(trainX, trainY, 0.9)

    print('Training Data shape')
    print(trainX.shape)
    print(trainY.shape)
    model = init_network(2,784,[128,64,10])
    # # train_history_1 = np.load('train_history.npy')
    # # val_history_1 = np.load('val_history.npy')
    # # val_acc_1 = np.load('val_acc.npy')
    # # train_acc_1 = np.load('train_acc.npy')
    # # loadModel(model,'bestmodel_task3_1000')
    model, train_history, val_history, train_acc, val_acc = train_sgd(model, trainX, trainY, valX, valY, learning_rate = 0.1, batch_size =100, training_epochs =10)
    # # train_history = np.append(train_history_1, train_history)
    # # val_history = np.append(val_history_1, val_history)
    # # val_acc = np.append(val_acc_1, val_acc)
    # # train_acc = np.append(train_acc_1, train_acc) 
    # #save the model 
    saveModel(model,'bestmodel_task3_3layer')
    # np.save('train_history', train_history)
    # np.save('val_history', val_history)
    # np.save('train_acc', train_acc)
    # np.save('val_acc', val_acc)
   
    
  
    # Loading Model
    mm = init_network(2,784,[128,64,10])
    loadModel(mm,'bestmodel_task3_3layer')

    # loading history
    # train_history = np.load('train_history.npy')
    # val_history = np.load('val_history.npy')
    # val_acc = np.load('val_acc.npy')
    # train_acc = np.load('train_acc.npy')

    # PLoting loss and accuracy 
    #plot_curves(train_history, val_history, 'Training vs Validation loss', 'Epoch', 'Loss')
    #plot_curves(train_acc, val_acc, 'Training vs Validation Accuracy', 'Epoch', 'Accuracy')

    ## Ploting Tsne plots
    # tsne_plot(mm,testX,testY, 1, 't-SNE plot (first hidden layer)')
    # tsne_plot(mm,testX,testY, 2, 't-SNE plot (second hidden layer)')
    # tsne_plot(mm,testX,testY, 3, 't-SNE plot (third hidden layer)')
 
    # check accuracy of that model
    #accuracy(mm,testX,testY,trainX,trainY)

    # #Plot confusion Matrix
    # confusion_matrix_plot(mm,testX,testY)
    # confusion_matrix_plot(mm,valX,valY)
    # confusion_matrix_plot(mm,trainX,trainY)

    
if __name__ == '__main__':
    main()