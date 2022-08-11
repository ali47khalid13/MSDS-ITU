# import all necessary libraries 
import numpy as np                                    # For matrices and MATLAB like functions                  
from sklearn.model_selection import train_test_split  # To split data into train and test set
# for plotting graphs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Viz_Data = False

def plot_predictions(testY, y_pred):
    plt.figure(figsize=(15,8))
    plt.plot(testY.squeeze(), linewidth=2 , label="True")
    plt.plot(y_pred.squeeze(), linestyle="--",  label="Predicted")
    plt.legend()
    plt.show()
    
def plot_losses(epoch_loss):
    plt.plot(epoch_loss, linewidth=2)
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.show()


# ### Data Generation
# 
# Input Feature Vector : X = [x_1 ,  x_2]^T 
# Target Variable : Y$ 
# Y = x_1^2 + x_2^3 + x_1x_2
def generate_data(n_samples=1500):

    X = np.random.uniform(-5,5, (n_samples, 2) ).astype(np.float32)

    Y = (X[:,0]**2 + X[:,1]**3 + X[:,0]*X[:, 1]).astype(np.float32)
    return X, Y


def visualize_data(X, Y):
    # ### Visualize Mapping from Input Feature Vector X = [x_1 ,  x_2]^T  to target variable Y
    fig = plt.figure(figsize=(20,10))
    ax = plt.axes(projection='3d')

    ax.plot_trisurf(X[:, 0],X[:, 1],Y, cmap='twilight_shifted')
    ax.view_init(30, -40)

    ax.set_xlabel(r'$x_1$', fontsize=20)
    ax.set_ylabel(r'$x_2$', fontsize=20)
    ax.set_zlabel(r'$y$'  , fontsize=20)
    plt.show()


# ---
# 
# ## Linear Regression with Numpy
# ### Forward and Backpropagation
# 
# We will implement linear regression. We will follow Pytorch style code pattern which will help in upcoming assigments.

class Network(object):        
    
    """
    We will implement Linear Regression Network as a python class. 
    Object of this class will have a few attributes for example learnable parameters
    and functions such as forward(), backward_propagate() etc.
    
    """
    
    # Initialize attributes of object
    def __init__(self,  n_features = 2):        
        
        # No. of input features
        self.n_features  = n_features
        
        
        # Learnable weights of neural network
        # initial random theta
        self.theta = np.array([np.random.rand() for i in range(self.n_features)])
        
    
    
    # This function just prints a few properties of object created from this class    
    def __str__(self):
        
        msg = "Linear Regression:\n\nSize of Input = " + str(self.n_features)
                
        return  msg
        
       
    
        
    # Forward propagation through neural network   
    def forward(self, x):
        
        y_hat =  np.dot(x, self.theta).reshape(len(x),1)
        
        return y_hat  

    
    # Backward propagation through neural network
    def backward(self , y_hat , x , y , lr):

        # Assuming, MSE (mean square error) loss has been computed
        # Using chain rule, find gradients of learnable parameters of neural network
        # Adjust weights and biases using gradients and learning rate 

        lr = lr
        
        batch_size = y_hat.shape[0]
        
        # Record gradients for all examples in given batch
        
        grad = np.array([(-2*np.dot(np.transpose(y-y_hat),x[:,i].reshape(len(x),1))/len(y))[0][0] for i in range(len(x[1]))])
                
        # Gradient Descent
        # Update theta using gradients
        
        self.theta -= lr*grad

            
    def loss(self, y, y_hat):
        # Mean Square Error Loss
        loss = np.dot(np.transpose(y-y_hat), y-y_hat)/ len(y)
        return loss
 

# ## Train Network using gradient descent
def train(model, n_epochs, lr, trainX, trainY, valX, valY):
    model = Network( n_features = 2  )
    print(model)
    n_examples = trainX.shape[0]
    epoch_loss = []
    print("\n\nTraining...")
    for epoch in range(n_epochs):
        loss=0
        y_hat = model.forward(trainX)
        model.backward( y_hat , trainX , trainY , lr)
        loss = model.loss(trainY,y_hat)
        epoch_loss.append(loss[0])
        # validation
        val_preds = model.forward(valX)
        #plot_predictions(valY, val_preds)
        print("Epoch = %3d   Loss = %3.3f"%(epoch, epoch_loss[epoch]) )
    print("\nDone.")    

    return model, epoch_loss

# Train using mini batch gradient descent
def train_batch(model, n_epochs, lr, trainX, trainY, valX, valY, batch_size):
    
    n_examples = trainX.shape[0]
    n_batches = n_examples//batch_size
    epoch_loss = []
    print("\n\nTraining...")
    for epoch in range(n_epochs):
        loss=0
        for i in range(n_batches):
            x = trainX[i*batch_size:(i+1)*batch_size]
            y = trainY[i*batch_size:(i+1)*batch_size] 
            y_hat = model.forward(x)
            model.backward( y_hat , x , y , lr)
            loss += model.loss(y,y_hat)[0]
        epoch_loss.append(loss.squeeze()/n_batches)
        # validation
        val_preds = model.forward(valX)
        # plot_predictions(valY, val_preds)   
        print("Epoch = %3d   Loss = %3.3f"%(epoch, epoch_loss[epoch]) )
    print("\nDone.")
    return model, epoch_loss




def main():
    X, Y = generate_data()
    if Viz_Data:
        visualize_data(X, Y)
    # Split the dataset into training and testing and validation
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=42)
    trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.2, random_state=42)  # 0.25 x 0.8 = 0.2

    trainY = trainY.reshape(trainY.shape[0], 1 )
    valY = valY.reshape(valX.shape[0], 1 )
    testY  = testY.reshape(testY.shape[0] ,1)
    print("Shape of Train Data:")
    print("TrainX: " , trainX.shape)
    print("TrainY: " , trainY.shape)

    print("\nShape of Test Data:")
    print("TestX: " , testX.shape)
    print("TestY: " , testY.shape)
    
    
    # ### Creating a Randomly Initialized Neural Network
    model = Network(n_features = 2)
    print(model)
    # ### Printing Randomly Initialized theta
    print("Theta = \n", model.theta)
    # Take 10 exampls to check if everything is working
    x = trainX[:10]
    y = trainY[:10]
    lr = 0.1
    y_hat = model.forward(x)
    print(y_hat.shape)
    print(y.shape)
    print("theta matrix before weight update:")
    print("\ntheta = \n", model.theta)
    model.backward( y_hat , x , y , lr)
    print("\n\ntheta matrix after weight update:")
    print("\ntheta = \n", model.theta)
    
    # ## Train Network using gradient descent
    model = Network(n_features = 2)
    lr = 0.001
    n_epochs=100
    model, epoch_loss = train(model, n_epochs, lr, trainX, trainY, valX, valY)
    plot_losses(epoch_loss)
    # ## Test prediction Prediction
    y_pred = model.forward(testX)
    plot_predictions(testY, y_pred)
    
    # Train using mini batch gradient descent
    model = Network( n_features = 2 )
    batch_size = 10
    lr = 0.001
    n_epochs=100
    model, epoch_loss = train_batch(model, n_epochs, lr, trainX, trainY, valX, valY, batch_size)
    plot_losses(epoch_loss)
    # ## Test prediction Prediction
    y_pred = model.forward(testX)
    plot_predictions(testY, y_pred)

    

if __name__ == '__main__':
    main()
    