"""
source: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

"""
import os
import time
import cv2
import numpy as np
import shutil
import sys
from numpy.core.fromnumeric import argmax

import torch
import torch.optim as optim
import torchvision
from dataset.BDD100k_lanes import BDDDataset
from model_oh import SCNN
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils.lr_scheduler import PolyLR
#from utils.transforms import *
import matplotlib.pyplot as plt
from tqdm import tqdm #progress bar
#import pdb pdb.set_trace() for debugging

# Directory settings
working_dir = "C:/Users/ynfuc/Documents/Masterarbeit/.vscode/BDD100k_implements/SCNN_Pytorch"
bdd100k_train_img_path = working_dir + "/dataset/images/t_train/"
bdd100k_train_dl_path = working_dir + "/dataset/lanes_gt3/t_train/"
bdd100k_val_img_path = working_dir + "/dataset/images/t_val/"
bdd100k_val_dl_path = working_dir + "/dataset/lanes_gt3/t_val/"
exp_dir = working_dir + "/experiments/exp13/lanes/"
exp_name = "t001"

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Use device: ", device)
torch.backends.cudnn.benchmark = True


#Data loader parameters
params = {"batch_size": 1, "shuffle": True, "num_workers": 4, "pin_memory": True}
max_epoch = 1000
resize_shape = tuple([512, 288])
optim_set = {"lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4, "nesterov": True}
lr_set = {"warmup": 200, "max_iter": 70000, "min_lrs": 1e-06}

# Define training dataset and data loader
train_bdd100k = BDDDataset(image_path=bdd100k_train_img_path, drivable_path = bdd100k_train_dl_path)
train_bdd100k_dataset_loader = DataLoader(dataset=train_bdd100k, **params)

# Define validation dataset and data loader
val_bdd100k = BDDDataset(image_path=bdd100k_val_img_path, drivable_path = bdd100k_val_dl_path)
val_bdd100k_dataset_loader = DataLoader(dataset=val_bdd100k, **params)

#Declare model & optimizers
net = SCNN(resize_shape, pretrained=True)
net = net.to(device)
#torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
#torch.cuda.set_device()
#net = torch.nn.parallel.DistributedDataParallel(net)
#net = torch.nn.DataParallel(net)
#
#net.eval()
tensorboard = SummaryWriter(exp_dir + "tb/")


optimizer = optim.SGD(net.parameters(), **optim_set)
lr_scheduler = PolyLR(optimizer, 0.9, **lr_set)
best_val_loss = 1000

#@profile
def train(epoch):
    print("Train Epoch: {}".format(epoch))
    net.train()
    train_loss = 0
    train_loss_seg = 0
    ##train_loss_exist = 0
    epoch_accuracy = 0

    progressbar = tqdm(range(len(train_bdd100k_dataset_loader)))
    #Training loop
    for batch_idx, sample in enumerate(train_bdd100k_dataset_loader):
        # move to GPU
        img = sample['image'].to(device)
        segLabel = sample['label'].to(device)

        #null gradient, get model output
        optimizer.zero_grad()
        seg_pred, exist_pred, loss_seg, loss_exist = net(img, segLabel) #, loss

        
        loss_seg = loss_seg.sum()
        loss_seg.requres_grad = True
        #loss_exist = loss_exist.sum()
        #loss = loss.sum()
        #loss.requres_grad = True

        #backprop, grad, learning rate update
        loss_seg.backward()
        optimizer.step()
        lr_scheduler.step()

        iter_idx = epoch * len(train_bdd100k_dataset_loader) + batch_idx
        #train_loss = loss.item()
        train_loss_seg = loss_seg.item()
        #train_loss_exist = loss_exist.item()

        #Calculate accuracy
        predicted = torch.argmax(seg_pred.data, dim=1) #returns sec arg of torch.max 
        correct_train = predicted.eq(segLabel.data).sum().item()
        accuracy = 100 * correct_train / segLabel.numel()
        
        #Save epoch accuracy in tensorboard
        epoch_accuracy +=accuracy
        if batch_idx >= (len(train_bdd100k_dataset_loader)-1):
            epoch_accuracy = epoch_accuracy/len(train_bdd100k_dataset_loader)
            tensorboard.add_scalar("accuracy", epoch_accuracy, epoch)

        progressbar.set_description("batch loss: {:.3f}".format(loss_seg.item()))
        progressbar.update(1)

        lr = optimizer.param_groups[0]['lr']
        tensorboard.add_scalar("train_loss", train_loss_seg, iter_idx)
        tensorboard.add_scalar("learning_rate", lr, iter_idx)
        """
        print("img size: ", img.size(0), "label size: ", segLabel.size(0))
        print("img size: ", type(img.size(0)), "label size: ", type(segLabel.size(0)))
        print("same: ", img.size(0)==segLabel.size(0), "diff: ", img.size(0)!=segLabel.size(0))
        """
        """
        #Test results
        test_pr = np.transpose(seg_pred[0].detach().numpy(),(1,2,0))
        test_gt = np.transpose(segLabel[0].detach().numpy(),(1,2,0))
        test_pr = test_pr * 255
        test_gt = test_gt * 255
        plt.imshow(test_pr)
        plt.pause(10)
        plt.imshow(test_gt)
        plt.pause(10)
        """
        #tensorboard.add_graph(net, input_to_model=img, verbose=False)


    progressbar.close()
    tensorboard.flush()

    #Save model & settings in exp_name.pth
    if epoch % 1 == 0:
        save_dict = {
            "epoch": epoch,
            "net": net.module.state_dict() if isinstance(net, torch.nn.DataParallel) else net.state_dict(),
            "optim": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "best_val_loss": best_val_loss
        }
        save_name = os.path.join(exp_dir, exp_name + '.pth')
        torch.save(save_dict, save_name)
        print("model is saved: {}".format(save_name))

    print("------------------------\n")

"""
    #average trainloss calc + print every 100 batches
    train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))         
    if batch_idx % 100 == 0:
        print('Epoch %d, Batch %d loss: %.6f' %(epoch, batch_idx + 1, train_loss))
                  """


def val(epoch):
    global best_val_loss
    net.eval()
    print("Val Epoch: {}".format(epoch))

    net.eval()
    val_loss = 0
    val_loss_seg = 0
    #val_loss_exist = 0 #CBE_loss not available for BDD100k
    progressbar = tqdm(range(len(val_bdd100k_dataset_loader)))

    #Validation
    with torch.set_grad_enabled(False):
        total_train = 0
        correct_train = 0
        epoch_accuracy = 0

        for batch_idx, sample in enumerate(val_bdd100k_dataset_loader):
            #Transfer to GPU
            img = sample['image'].to(device)
            segLabel = sample['label'].to(device)
            #exist = sample['exist'].cuda()
            #local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            seg_pred, exist_pred, loss_seg, loss_exist = net(img, segLabel)

            loss_seg = loss_seg.sum()
                #loss_exist = loss_exist.sum()
            #loss = loss.sum()

            """
            predicted = torch.argmax(seg_pred.data, dim=1) #returns sec arg of torch.max
            #print(total_train, predicted.shape, segLabel.shape)   
            correct_train = predicted.eq(segLabel.data).sum().item()
            accuracy = 100 * correct_train / segLabel.numel()
            predict = predicted.eq(segLabel) #True/False übereinstimmung
            np.set_printoptions(threshold=sys.maxsize)
            #print("Variante1: {:.3f}".format(accuracy))
            epoch_accuracy +=accuracy
            if batch_idx >= (len(train_bdd100k_dataset_loader)-1):
                epoch_accuracy = epoch_accuracy/len(val_bdd100k_dataset_loader)
                tensorboard.add_scalar("val_accuracy", epoch_accuracy, epoch)
            """
            predicted = torch.argmax(seg_pred.data, dim=1) #returns sec arg of torch.max
            #print(total_train, predicted.shape, segLabel.shape)   
            correct_train = predicted.eq(segLabel.data).sum().item()
            accuracy = 100 * correct_train / segLabel.numel()
            #predict = predicted.eq(segLabel) #True/False übereinstimmung
            np.set_printoptions(threshold=sys.maxsize)
            #print("Variante1: {:.3f}".format(accuracy))
            epoch_accuracy +=accuracy
            if batch_idx >= (len(train_bdd100k_dataset_loader)-1):
                epoch_accuracy = epoch_accuracy/len(val_bdd100k_dataset_loader)
                tensorboard.add_scalar("val_accuracy", epoch_accuracy, epoch)
            #print( predicted.shape, segLabel.shape)
            """
            https://www.kaggle.com/devilsknight/malaria-detection-with-pytorch
            # convert output probabilities to predicted class
            pred = output.data.max(1, keepdim=True)[1]
            # compare predictions to true label
            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            total += data.size(0)
        """
            
            """
            #prediction plot
            #print("Label: ",segLabel[0,110:190,210:290].numpy())
            #print("predic: ",predicted[0,110:190,210:290].numpy())
            #print("Compare: ",predict[0,110:190,210:290].numpy())
            print (seg_pred.shape)
            f = plt.figure()
            f.add_subplot(2,2,1)
            plt.imshow(img[0].permute(1,2,0))
            f.add_subplot(2,2,2)
            plt.imshow(segLabel[0])
            f.add_subplot(2,2,3)
            plt.imshow(predicted[0])
            f.add_subplot(2,2,4)
            plt.imshow(predict[0].detach().cpu())
            plt.show(block=True)
            #plt.pause(5)
            """
            if batch_idx ==0:
                
                #val_images = [img[0].permute(1,2,0), segLabel[0], predicted[0], predict[0].detach().cpu()]
                tensorboard.add_image("Image: ", img[0], global_step=epoch, dataformats='CHW')
                """
                tensorboard.add_image("Image_gt: ", segLabel[0], global_step=epoch, dataformats='HW')
                tensorboard.add_image("Image_predicted: ", predicted[0], global_step=epoch, dataformats='HW')
                tensorboard.add_image("Image_compare: ", predict[0].detach().cpu(), global_step=epoch, dataformats='HW')
                """
                ##img_grid = torchvision.utils.make_grid([segLabel[0].squeeze(), predicted[0]])
                ##tensorboard.add_image("Val_Images: ", img_grid, global_step=epoch, dataformats='CHW')

            # visualize validation every 5 frame, 50 frames in all
            #gap_num = 25
            #if batch_idx%gap_num == 0 and batch_idx < 50 * gap_num:
            origin_imgs = []
            #seg_pred = seg_pred.detach().cpu().numpy()
            if batch_idx ==0:
                #np.set_printoptions(threshold=sys.maxsize)
                #print(net)
                #print(segLabel[0,180:230,450:512].numpy())
                #print(seg_pred[0,0,180:230,450:512].numpy())
                #plt.imshow(predict[0].detach().cpu())
                #plt.pause(5)
                """
                #plt.subplot(1, 6, 1)
                plt.imshow(segLabel[0])
                #plt.pause(5)

                for i in range (0, 3):
                    plt.figure()
                    print(seg_pred.shape)
                    plt.imshow(torch.exp(seg_pred[0, i, :, :]).detach().numpy())
                    plt.pause(5)
                    
                plt.close()
                """
                    #exist_pred = exist_pred.detach().cpu().numpy()
                           
            #val_loss += loss.item()
            val_loss_seg += loss_seg.item()
            #val_loss_exist += loss_exist.item()

            progressbar.set_description("batch loss: {:.3f}".format(loss_seg.item()))
            progressbar.update(1)

    progressbar.close()
    iter_idx = (epoch + 1) * len(train_bdd100k_dataset_loader)  # keep align with training process iter_idx
    #tensorboard.add_scalar("val_loss", val_loss, iter_idx)
    tensorboard.add_scalar("val_loss_seg", val_loss_seg, iter_idx)
    #tensorboard.scalar_summary("val_loss_exist", val_loss_exist, iter_idx)
    tensorboard.flush()
    

    print("------------------------\n")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_name = os.path.join(exp_dir, exp_name + '.pth')
        copy_name = os.path.join(exp_dir, exp_name + '_best.pth')
        shutil.copyfile(save_name, copy_name)


            #Model computions

def main():
    global best_val_loss
    resume = False
    if resume:
        save_dict = torch.load(os.path.join(exp_dir, exp_name + '.pth'))
        if isinstance(net, torch.nn.DataParallel):
            net.module.load_state_dict(save_dict['net'])
        else:
            net.load_state_dict(save_dict['net'])
        optimizer.load_state_dict(save_dict['optim'])
        lr_scheduler.load_state_dict(save_dict['lr_scheduler'])
        start_epoch = save_dict['epoch'] + 1
        best_val_loss = save_dict.get("best_val_loss", 1e6)
    else:
        start_epoch = 0

    for epoch in range (start_epoch, max_epoch):
        train(epoch)
        #val(epoch)
        if epoch % 1 == 0:
            print("\nValidation For Experiment: ", exp_dir)
            print(time.strftime('%H:%M:%S', time.localtime()))
            val(epoch)


if __name__ == "__main__":
    main()


"""
            probs = torch.log_softmax(seg_pred, dim = 1)
            _, tags = torch.max(probs, dim = 1)
            corrects = torch.eq(tags,segLabel).int()
            acc = corrects.sum()/corrects.numel()
            acc = acc * 100
            print("Variante2: ",float(acc))
            """

#for images, labels in train_bdd100k_dataset_loader:
   #Feed the data to the model