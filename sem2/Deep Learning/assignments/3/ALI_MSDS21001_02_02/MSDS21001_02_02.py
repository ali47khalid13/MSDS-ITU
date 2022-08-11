import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import itertools
import numpy as np

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torch import nn
from torch import optim

import PIL
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, recall_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

class Custom_Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None): 
        # Run once
        self.img_labels = pd.read_csv(csv_file)
        self.transform = transform
        self.img_dir = img_dir
    
    def __len__(self):
        # return the number of samples in dataset
        return len(self.img_labels)

    def __getitem__(self, idx):
        # loads and returns a sample from the dataset at the given index
        image = os.path.join(self.img_dir ,self.img_labels.iloc[idx, 0])
        image = PIL.Image.open(image)
        label = torch.tensor(self.img_labels.iloc[idx, 1])
    
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_dataset(dataset_path, train_size, validation_size, batch_size):

    train_img = os.path.join(dataset_path, 'train_new')
    test_img = os.path.join(dataset_path, 'test_new')
    train_csv = os.path.join(dataset_path, 'train_new.csv')
    test_csv = os.path.join(dataset_path, 'test_new.csv')
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(0,0.5), 
                                    transforms.Lambda(lambda x: x.repeat(3,1,1))])
    # read dataset
    train_data = Custom_Dataset(csv_file=train_csv, img_dir=train_img, transform = transform)
    test_data = Custom_Dataset(csv_file=test_csv, img_dir=test_img, transform = transform)

    # specify sizes
    train_set_size = int(len(train_data) * train_size)
    val_set_size = len(train_data) - train_set_size    

    # split data
    train_data, val_data = torch.utils.data.random_split(train_data, [train_set_size,val_set_size])

    # create dataloader for each data (test,train,val)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    #return test_data
    return train_data_loader, val_data_loader, test_data_loader

class init_network(nn.Module):

    def conv_block(self,in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def convDw_block(self,in_channels, out_channels, stride):
        return nn.Sequential(
            #depth wise convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            #point wise convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            )
    
    def __init__(self,conv_block_count, mobilenet_block_count, input_dim, num_classes):
        super(init_network,self).__init__()
        self.net = nn.Sequential()
        self.in_channels = input_dim[0]
        self.out_channels = 32
        self.stride = 2
        for i in range(conv_block_count):
            if i%2 ==0:
                self.stride = 2
            else:
                self.stride = 1
            self.net.add_module("conv_block "+str(i+1), self.conv_block(self.in_channels,self.out_channels, stride =self.stride ))
            self.in_channels = self.out_channels
            self.out_channels *=2
        for i in range(mobilenet_block_count):
            if i%2 ==0:
                self.stride = 1
            else:
                self.stride = 2
            self.net.add_module("convDw_block "+str(i+1), self.convDw_block(self.in_channels,self.out_channels, stride =self.stride))
            self.in_channels = self.out_channels
            self.out_channels *=2
        self.net.add_module("pooling_layer ", nn.AvgPool2d(7))
        
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

def train(net,  train_data_loader, val_data_loader, lr, epochs, loss_func, optimizer, device):

    learning_rate = lr
    optimizer = optimizer(net.parameters(), lr=learning_rate)
    train_step = len(train_data_loader)
    val_step = len(val_data_loader)
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    early_stoping_flag = 0
    learning_rate_decay_flag = 0 
    EPS = 1e-6

   

    for epoch in range(epochs): # iterate over epochs

        if epoch >=2:
            if train_loss[-2] - train_loss[-1] <EPS:
                learning_rate_decay_flag+=1
            else:
                learning_rate_decay_flag =0
            if val_loss[-1] - val_loss[-2] >EPS and train_loss[-2] - train_loss[-1] >EPS :
                early_stoping_flag+=1
            else:
                early_stoping_flag =0
        
        if learning_rate_decay_flag >=3:
            learning_rate = learning_rate/10
            optimizer = optimizer(net.parameters(), lr=learning_rate)
            learning_rate_decay_flag =0
        
        if early_stoping_flag >=3:
            return net, [train_loss, train_acc, val_loss, val_acc]

        t_loss = 0
        t_acc = 0 
        for i, data in enumerate(train_data_loader): # iterate over batches
            # get image and labels data is in tuple form (inputs, label)
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero-out gradients
            optimizer.zero_grad()
            # inputs = inputs.view(inputs.shape[0],-1)
            net.train()
            torch.enable_grad()     
            outputs = net(inputs)
            _, pred = torch.max(outputs, 1)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            t_acc += torch.sum(pred == labels)/len(labels)

            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, epochs, i+1, train_step, loss.item()))
    
        v_loss = 0
        v_acc = 0
        for i, data in enumerate(val_data_loader): # iterate over batches
            # get image and labels data is in tuple form (inputs, label)
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # inputs = inputs.view(inputs.shape[0],-1)
            net.eval()
            torch.no_grad()    
            outputs = net(inputs)
            _, pred = torch.max(outputs, 1)
            loss = loss_func(outputs, labels)
            v_loss += loss.item()
            v_acc += torch.sum(pred == labels)/len(labels)
     
        train_loss.append(t_loss/train_step)
        train_acc.append(t_acc/train_step)
        val_loss.append(v_loss/val_step)
        val_acc.append(v_acc/val_step)
        print ('Epoch [{}/{}], train_loss: {:.4f}, val_loss: {:.4f}' .format(epoch+1, epochs, train_loss[-1], val_loss[-1]))
    return net, [train_loss, train_acc, val_loss, val_acc]

# Function to test the model
def test(net, test_data_loader, device):
    net = net.to(device)
    test_steps = len(test_data_loader)
    t_acc=0
    labels_all = torch.Tensor()
    pred_all = torch.Tensor()
    output_all = torch.Tensor()
    for i, data in enumerate(test_data_loader): # iterate over batches
    # get image and labels data is in tuple form (inputs, label)
        inputs, labels = data
        labels_all = torch.cat((labels_all,labels))
        inputs = inputs.to(device)
        labels = labels.to(device)
        # inputs = inputs.view(inputs.shape[0],-1)
        net.eval()
        torch.no_grad()   
        outputs = net(inputs)
        _, pred = torch.max(outputs, 1)
        pred_all = torch.cat((pred_all,pred))
        output_all = torch.cat((output_all,outputs),0)
        t_acc += torch.sum(pred == labels)/len(labels)
    print('Test Accuracy: {:.4f}'.format(t_acc/test_steps))
    return pred_all, labels_all, output_all

def tsne_plot(lables,activations, title):
    tsne = TSNE(n_components=2, verbose=0, random_state=123) 
    z = tsne.fit_transform(activations.data)
    df = pd.DataFrame()
    df["y"] = lables
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),data=df,palette=sns.color_palette("hls", 10))
    plt.title(title)
    plt.show()

def conv_layers_weights(net):
    model_weights =[]
    conv_layers = []
    model_children = list(net.children())
    counter = 0
    #append all the conv layers and their respective wights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter+=1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                if type(model_children[i][j]) == nn.Conv2d:
                    counter+=1
                    model_weights.append(model_children[i][j].weight)
                    conv_layers.append(model_children[i][j])
                elif type(model_children[i][j]) == nn.Sequential:
                    for k in range(len(model_children[i][j])):
                        if type(model_children[i][j][k]) == nn.Conv2d:
                            counter+=1
                            model_weights.append(model_children[i][j][k].weight)
                            conv_layers.append(model_children[i][j][k])
    print(f"Total convolution layers: {counter}")
    return model_weights, conv_layers

def plot_weights(net):
    images = []
    model_weights, conv_layers = conv_layers_weights(net)
    #print('length',model_weights.shape)
    last_layer_weights = model_weights[-1]
    print(len(last_layer_weights))
    for i in range(len(last_layer_weights)):
        for j in range(len(last_layer_weights[i])):
            images.append(last_layer_weights[i][j].data.reshape(3,3))

    columns = 16
    rows = 16
    fig, ax = plt.subplots(nrows=rows, ncols=columns, sharex=True, sharey=True,figsize=(columns*2, rows*2))

    for i in range(rows):
        for j in range(columns):
            ax[i,j].imshow(images[i])
    plt.show()

def wrong_pred(net, test_data_loader, device):
    net = net.to(device)
    test_steps = len(test_data_loader)
    t_acc=0
    wrong = 0 
    wrong_pred = []
    wrong_label = []
    for i, data in enumerate(test_data_loader): # iterate over batches
    # get image and labels data is in tuple form (inputs, label)
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        net.eval()
        torch.no_grad()   
        outputs = net(inputs)
        _, pred = torch.max(outputs, 1)        
        t_acc = torch.sum(pred == labels)

        if t_acc <60:
            plt.figure(figsize=(15, 12))
            plt.subplots_adjust(hspace=0.5)
            plt.suptitle("wrong predictions", fontsize=18, y=0.95)
            print('labels and predictions of wrong prediction')
            for i in range(0,64):
                if pred[i] != labels[i] and wrong <5:
                    wrong_label.append(labels[i])
                    wrong_pred.append(pred[i])
                    wrong += 1
                    plt.subplot(1, 4, wrong)
                    # filter df and plot ticker on the new subplot axis
                    plt.imshow(inputs[i].permute(1, 2, 0))
                if wrong ==4:
                    print('labels :', wrong_label)
                    print('prediction :', wrong_pred)
                    plt.show()
                    return 1

def right_pred(net, test_data_loader, device):
    net = net.to(device)
    test_steps = len(test_data_loader)
    t_acc=0
    right = 0 
    right_pred = []
    right_label = []
    for i, data in enumerate(test_data_loader): # iterate over batches
    # get image and labels data is in tuple form (inputs, label)
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        net.eval()
        torch.no_grad()   
        outputs = net(inputs)
        _, pred = torch.max(outputs, 1)        
        t_acc = torch.sum(pred == labels)

        if t_acc>3:
            plt.figure(figsize=(15, 12))
            plt.subplots_adjust(hspace=0.5)
            plt.suptitle("right predictions", fontsize=18, y=0.95)
            print('labels and prediction of right predictions')
            for i in range(0,64):
                if pred[i] == labels[i] and right <5:
                    right_label.append(labels[i])
                    right_pred.append(pred[i])
                    right += 1
                    plt.subplot(1, 4, right)
                    # filter df and plot ticker on the new subplot axis
                    plt.imshow(inputs[i].permute(1, 2, 0))
                if right ==4:
                    print('labels :', right_label)
                    print('prediction :', right_pred)
                    plt.show()
                    return 1

def visualize(train_loss, train_acc, val_loss, val_acc, test_pred, test_labels):

    # visualizing the training and validation loss
    plt.plot(train_loss, linewidth=2, label='train')
    plt.plot(val_loss, linewidth=2, label='validation')
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.legend()
    plt.title('training and validation loss')
    plt.show()
    # visualizing the training and validation accuracy
    plt.plot(train_acc, linewidth=2, label='train')
    plt.plot(val_acc, linewidth=2, label='validation')
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.legend()
    plt.title('training and validation Accuracy')
    plt.show()
    
    test_labels = np.array(test_labels)
    test_pred = np.array(test_pred)
    cm = confusion_matrix(test_pred, test_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    print('Test Set Accuracy: ', accuracy_score(test_pred, test_labels))    
    print('Test Set Recall Score: ', recall_score(test_pred, test_labels, average='weighted'))
    print('Test Set F1 Score: ', f1_score(test_pred, test_labels, average='weighted')) 

    y_test = label_binarize(test_labels, classes=list(range(10)))
    y_score = label_binarize(test_pred, classes=list(range(10)))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(10):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = itertools.cycle(["aqua", "darkorange", "cornflowerblue", 'green', 'blue','cyan','red','yellow','purple','pink'])
    for i, color in zip(range(10), colors):
        plt.plot(fpr[i],tpr[i],color=color, lw=2, label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Some extension of Receiver operating characteristic to multiclass")
    plt.legend(loc="lower right")
    plt.show()


def plot_examples(test_loader):
    def imshow(img):
        img = img * 0.5
        np_img = img.numpy() 
        plt.imshow(np.transpose(np_img*255, (1,2,0)).astype(np.uint8))
        plt.show()
    dataiter = iter(test_loader)
    images, test_labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))

    # show labels
    print(test_labels)


def save_net(net):
    torch.save(net, 'saved_models/task1/best_model.pkl')

def load_net():
    net = torch.load('saved_models/task2/e1/best_model.pkl')
    net.eval()
    return net


def main(dataset_path, train_size, val_size, train_epochs, conv_block_count, mobilenet_block_count,num_classes
    , loss, optimizer, batch_size, learning_rate, GPU=False, Training=False, Visualize=False):
  
    # setting the device
    device = torch.device('cuda' if (GPU == True and torch.cuda.is_available()) else 'cpu')

    # calling the loadDataset function and fetching the train, validation and test loaders
    train_data_loader, val_data_loader, test_data_loader = load_dataset(dataset_path, train_size, val_size, batch_size)

    # finding input dimentions
    dataiter = iter(test_data_loader)
    train_data, train_labels = dataiter.next()

    # calling the init_network function to define the architecture of neural network
    net = init_network(conv_block_count, mobilenet_block_count, input_dim, num_classes).to(device)

    if Training==True:
    
        # calling the train function to train the neural network on the images
        net, history = train(net, train_data_loader, val_data_loader, learning_rate, train_epochs, loss, optimizer, device)
        save_net(net)
        np.save('saved_models/task1/history', history)
    
    net = load_net() 
    history = np.load('saved_models/task2/e1/history.npy')

    # calling the test function to test the model
    pred, labels, outputs= test(net, test_data_loader, device)
    right_pred(net, test_data_loader, device)
    wrong_pred(net, test_data_loader, device)
    plot_examples(test_data_loader)
    tsne_plot(labels, outputs , 'Tsne Plot')

    if (Visualize==True):# and Training==True):
        # calling the visualization function for some visualizations
        visualize(history[0], history[1], history[2], history[3], pred, labels)
        plot_weights(net)

if __name__ == '__main__':

    print('Name: Ali Khalid')
    print('Roll Number: MSDS21001')
    
    dataset_path =  'dataset'
    batch_size =  64
    train_epochs =  1
    learning_rate = 0.0001
    # loss function
    loss =  nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam
    conv_block_count = 4
    mobilenet_block_count = 0
    input_dim = (3,28,28)
    num_classes = 10
    main(dataset_path, 0.8, 0.2, train_epochs, conv_block_count, mobilenet_block_count,num_classes
    , loss, optimizer, batch_size, learning_rate, GPU=False, Training=True, Visualize=True)