import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import itertools
import numpy as np
import torchvision.utils as vutils
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, recall_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from matplotlib import image as img
import glob
from torchvision import transforms, datasets

def load_dataset(dataset_path, batch_size, train_size):

    transform_train= transforms.Compose([#transforms.Grayscale(), 
                                         transforms.ToTensor(),
                                         #transforms.Resize((28,28)), 
                                         transforms.Normalize([0,0,0],[0.5, 0.5, 0.5])
                                         ])
    transform_test= transforms.Compose([#transforms.Grayscale(), 
                                         transforms.ToTensor(),
                                         #transforms.Resize((28,28)), 
                                         transforms.Normalize([0,0,0],[0.5, 0.5, 0.5])
                                         ])
    
    # read dataset
    train_data_path = os.path.join(dataset_path, "train")
    test_data_path = os.path.join(dataset_path, "test")
    train_dataset = datasets.ImageFolder(train_data_path, transform=transform_train)
    test_data = datasets.ImageFolder(test_data_path, transform=transform_test)

    train_set_size = int(len(train_dataset) * train_size)
    val_set_size = len(train_dataset) - train_set_size   
    train_data, val_data = torch.utils.data.random_split(train_dataset, [train_set_size,val_set_size])


    # create dataloader for each data (test,train,val)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    #return test_data
    return train_data_loader,val_data_loader, test_data_loader

## image size = 3 x 32 x 32
# Generator Code
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(    
            nn.ConvTranspose2d(100, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( 64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 3, 4,  stride=2, padding=1),
            nn.Tanh()      
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.cnn_layers_2 =  nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid(),
            )

        self.classification_layer =  nn.Sequential(
            nn.Linear(128*4*4, 1024),
            nn.Linear(1024, 10),
            )

    def forward(self, x):
        a = self.cnn_layers(x)
        x = self.cnn_layers_2(a)
        a = a.view(a.shape[0], -1)
        y = self.classification_layer(a)
        return x ,y


def train(num_epochs,train_data_loader,val_data_loader,netD,netG,device,real_label,fake_label,criterion,criterion1,optimizerD,optimizerG,fixed_noise):
    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    iters = 0
    train_step = len(train_data_loader)
    val_step = len(val_data_loader)

    print("Starting Training Loop...")

    for epoch in range(num_epochs):
        t_loss = 0
        t_acc = 0 
        v_loss = 0
        v_acc = 0 
        for i, data in enumerate(train_data_loader, 0):
            
            # diccriminator update 
            # Train with all-real batch
            netD.zero_grad()
            batch, labels1 = data
            batch = batch.to(device)
            labels = torch.eye(10)[labels1].to(device)
            batch_size = batch.size(0)
            label_real = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            output_real, labels_predicted_real = netD(batch)
            output_real = output_real.view(-1)
            # labels_predicted_real = labels_predicted_real
            errD = criterion(output_real, label_real)
            errC= criterion1(labels_predicted_real, labels)
            _, pred = torch.max(labels_predicted_real, 1)
            t_loss += errC.item()
            t_acc += torch.sum(pred == labels1)/len(labels1)
            errD_real = errD + errC
            D_x = output_real.mean().item()

            # Train with all-fake batch
            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake = netG(noise)
            label_fake = torch.full((batch_size,), fake_label, dtype=torch.float, device=device)
            output_fake, _ = netD(fake.detach())
            output_fake = output_fake.view(-1)
            errD_fake = criterion(output_fake, label_fake)
            #errD_fake.backward()
            D_G_z1 = output_fake.mean().item()
            errD = errD_real + errD_fake
            # Update D
            errD.backward()
            optimizerD.step()

            # generator update
            netG.zero_grad()
            label=torch.full((batch_size,), real_label, dtype=torch.float, device=device) # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output,_ = netD(fake)
            output = output.view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if(i%100==0):
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (epoch, num_epochs, i, len(train_data_loader),errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            iters += 1
            
        for i, data in enumerate(val_data_loader, 0):
            
            # diccriminator update 
            # Train with all-real batch
            netD.zero_grad()
            batch, labels1 = data
            batch = batch.to(device)
            labels = torch.eye(10)[labels1].to(device)
            batch_size = batch.size(0)
            output_real, labels_predicted_real = netD(batch)
            errC= criterion1(labels_predicted_real, labels)
            _, pred = torch.max(labels_predicted_real, 1)
            v_loss += errC.item()
            v_acc += torch.sum(pred == labels1)/len(labels1)

        train_loss.append(t_loss/train_step)
        train_acc.append(t_acc/train_step)
        val_loss.append(v_loss/val_step)
        val_acc.append(v_acc/val_step)
    
        #Check how the generator is doing by saving G's output on fixed_noise
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    return [G_losses, D_losses, img_list,train_loss, train_acc, val_loss, val_acc]


def save_net(netD, netG):
    torch.save(netD, 'best_model_netD.pkl')
    torch.save(netG, 'best_model_netG.pkl')

def load_net():
    netD = torch.load('best_model_netD.pkl')
    netG = torch.load('best_model_netG.pkl')
    netD.eval()
    netG.eval()
    return netD, netG

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

def transpose_conv_layers_weights(net):
    model_weights =[]
    conv_layers = []
    model_children = list(net.children())
    counter = 0
    #append all the conv layers and their respective wights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.ConvTranspose2d:
            counter+=1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                if type(model_children[i][j]) == nn.ConvTranspose2d:
                    counter+=1
                    model_weights.append(model_children[i][j].weight)
                    conv_layers.append(model_children[i][j])
                elif type(model_children[i][j]) == nn.Sequential:
                    for k in range(len(model_children[i][j])):
                        if type(model_children[i][j][k]) == nn.ConvTranspose2d:
                            counter+=1
                            model_weights.append(model_children[i][j][k].weight)
                            conv_layers.append(model_children[i][j][k])
    print(f"Total transpose convolution layers: {counter}")
    return model_weights, conv_layers

def plot_weights_conv(net,name):
    images = []
    model_weights, conv_layers = conv_layers_weights(net)
    #print('length',model_weights.shape)
    last_layer_weights = model_weights[-1]
    for i in range(len(last_layer_weights)):
        for j in range(len(last_layer_weights[i])):
            images.append(last_layer_weights[i][j].data.reshape(4,4))

    columns = 16
    rows = 16
    fig, ax = plt.subplots(nrows=rows, ncols=columns, sharex=True, sharey=True,figsize=(columns*2, rows*2))

    for i in range(rows):
        for j in range(columns):
            ax[i,j].imshow(images[i])
    plt.savefig(name)

def plot_weights_t_conv(net,name):
    images = []
    model_weights, conv_layers = transpose_conv_layers_weights(net)
    #print('length',model_weights.shape)
    last_layer_weights = model_weights[-1]
    for i in range(len(last_layer_weights)):
        for j in range(len(last_layer_weights[i])):
            images.append(last_layer_weights[i][j].data.reshape(4,4))

    columns = 16
    rows = 16
    fig, ax = plt.subplots(nrows=rows, ncols=columns, sharex=True, sharey=True,figsize=(columns*2, rows*2))

    for i in range(rows):
        for j in range(columns):
            ax[i,j].imshow(images[i])
    plt.savefig(name)

def main(dataset_path, train_epochs, loss,loss1, optimizer, batch_size, learning_rate, device,train_size, Training=False, Visualize=False):   

    train_data_loader, val_data_loader, test_data_loader =  load_dataset(dataset_path,batch_size,train_size)
    netD = Discriminator()
    netG = Generator()

    criterion = loss()
    criterion1 = loss1()

    fixed_noise = torch.randn(batch_size, 100, 1, 1)

    real_label = 1.
    fake_label = 0.

    optimizerD = optimizer(netD.parameters(), lr=learning_rate)
    optimizerG = optimizer(netG.parameters(), lr=learning_rate)

    if Training==True:
        
        # calling the train function to train the neural network on the images
        history =  train(train_epochs,train_data_loader,val_data_loader, netD,netG,device,real_label,fake_label,criterion,criterion1,optimizerD,optimizerG,fixed_noise)
        save_net(netD, netG)
        np.save('history', history)
    
    if Visualize ==True:

        netD, netG = load_net() 
        history = np.load('history.npy',allow_pickle=True)
        plot_weights_conv(netD, 'discriminator weights.png')
        plot_weights_t_conv(netG, 'Generator weights.png')
        # plt.figure(figsize=(10,5))
        # plt.title("Generator Loss During Training")
        # plt.plot(history[0],label="GGenerator")
        # plt.xlabel("iterations")
        # plt.ylabel("Loss")
        # plt.legend()
        # plt.savefig('Generator Loss.png')

        # plt.figure(figsize=(10,5))
        # plt.title("Discriminator Loss During Training")
        # plt.plot(history[1],label="DDiscriminator")
        # plt.xlabel("iterations")
        # plt.ylabel("Loss")
        # plt.legend()
        # plt.savefig('Discriminator Loss.png')

        # real_batch = next(iter(test_data_loader))

        # # Plot the real images
        # plt.figure(figsize=(15,15))
        # plt.subplot(1,2,1)
        # plt.axis("off")
        # plt.title("Real Images")
        # plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

        # # Plot the fake images from the last epoch
        # plt.subplot(1,2,2)
        # plt.axis("off")
        # plt.title("Fake Images")
        # plt.imshow(np.transpose(history[2][-1],(1,2,0)))
        # plt.savefig('images.png')

        # plt.figure(figsize=(10,5))
        # plt.plot(history[3], linewidth=2, label='train')
        # plt.plot(history[5], linewidth=2, label='validation')
        # plt.xlabel('Epoch', fontsize=15)
        # plt.ylabel('Loss', fontsize=15)
        # plt.legend()
        # plt.title('training and validation loss')
        # plt.savefig('t2_classification_loss.png')

        # plt.figure(figsize=(10,5))
        # plt.plot(history[4], linewidth=2, label='train')
        # plt.plot(history[6], linewidth=2, label='validation')
        # plt.xlabel('Epoch', fontsize=15)
        # plt.ylabel('Accuracy', fontsize=15)
        # plt.legend()
        # plt.title('training and validation Accuracy')
        # plt.savefig('t2_classification_acc.png')

if __name__ == '__main__':
    
    print('Name: Ali Khalid')
    print('Roll Number: MSDS21001')
    
    device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
    dataset_path =  '../data'
    batch_size =  64
    train_epochs =  1
    learning_rate = 0.0001
    train_size = 0.8
    loss =  nn.BCELoss
    loss1 = nn.CrossEntropyLoss
    optimizer = optim.Adam
    main(dataset_path, train_epochs, loss,loss1, optimizer, batch_size, learning_rate, device, train_size,Training=True, Visualize=True)