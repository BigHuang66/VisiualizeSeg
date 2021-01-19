import time
import copy
from visdom import Visdom
from torch.autograd import Variable
from models.my_model import *
from datasets.get_data import *

def train_model(model, criterion, optimizer, scheduler, num_epochs=50, visdom_flag=False):
    since = time.time()
    if visdom_flag == True:
        viz = Visdom()
        viz.line([[0., 0.]], [0], win='loss', opts=dict(title='Loss', legend=['train_loss', 'val_loss']))
        viz.line([[0., 0.]], [0], win='acc', opts=dict(title='Acc', legend=['train_acc', 'val_acc']))
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
 
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    loss, indices = torch.sort(loss, dim=0, descending=True)
                    loss = loss.mean()
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
                train_loss = running_loss / dataset_sizes[phase]
                train_acc = running_corrects.double() / dataset_sizes[phase]
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, train_loss, train_acc))
            else:
                val_loss = running_loss / dataset_sizes[phase]
                val_acc = running_corrects.double() / dataset_sizes[phase]
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, val_loss, val_acc))


            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = copy.deepcopy(model.state_dict())

             
        
        if visdom_flag == True:
            viz.line([[train_loss, val_loss]], [epoch], win='loss', update='append')
            viz.line([[train_acc.cpu().double(), val_acc.cpu().double()]], [epoch], win='acc', update='append')

    time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(100 * best_acc / len(val_dataset)))
    # str = 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60) + '    Best val Acc: {:4f}'.format(100 * best_acc / len(val_dataset))
    # with open('logs.txt', 'w') as f:  # 设置文件对象
    #     f.write(str)  # 将字符串写入文件中

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, './checkpoints/Alexnet.pth')
    return model