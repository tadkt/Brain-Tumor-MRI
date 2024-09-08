import torch
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F
import os
import numpy as np
import copy
import time
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.init as init

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out = F.softmax(out, dim=1)
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)

def train(model, criterion, optimizer, trainloader, testloader, epochs=2, log_interval=50, lr=0.1):
    print("-------Train Start-------")
    loss_history =[]
    train_acc_history = []
    test_acc_history = []
    best_acc = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for step, (batch_x, batch_y) in enumerate(trainloader):
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda() # Input data and label are moved to the GPU

            output = model(batch_x) # Process the input data
            
            optimizer.zero_grad() # Optimizer's gradients are zeroed
            loss = criterion(output, batch_y) # Loss based on (criterion) variable
            loss.backward() # Backpropagation
            optimizer.step() # Update the model parameters

            running_loss += loss.item() # Accumulate the loss
            loss_history.append(loss.item()) # Store the loss

            if step % log_interval == (log_interval-1):
                print('[%d, %5d] loss :.4f' %
                      (epoch + 1, step+1, running_loss/log_interval))
                running_loss = 0.0
            model.eval() # Use model.eval() before running model on validation set, then switch back
            print("-------Test Start-------") # later with model.train() when resuming on train set
            train_acc_history.append(test(model, trainloader))
            test_acc_history.append(test(model, testloader))
            if test_acc_history[-1]>best_acc:
                best_acc = test_acc_history[-1]
                best_epoch = epoch+1
                print("Saving...")
                state = model.state_dict() # Returns a dictionary containing a whole state of the module
                if not os.path.isdir('checkpoint'): # If the directory does not exist, create it
                    os.mkdir('checkpoint')
                torch.save(state, f'./checkpoint/ckpt_{lr:.4f}.pth') # Save the model state dictionary to a file
    print("-------Train End-------")
    return {
        'loss_history': loss_history,
        'train_acc_history': train_acc_history,
        'test_acc_history': test_acc_history,
        'best_acc': best_acc,
        'best_epoch': best_epoch
    }


def test(model, testloader):
    correct = 0
    total = 0

    with torch.no_grad():
        for test_x, test_y in testloader:
            images, labels = test_x.cuda(), test_y.cuda()
            output = model(images)
            # print(output)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

def PSO_train(models, criterion, epochs, Wmax, Wmin, c1, c2, train_loader, test_loader):
    '''
    :param models:
    :param epochs:
    :param Wmax:
    :param Wmin:
    :param c1:
    :param c2:
    :param trainloader:
    :param testloader:
    :return:
    '''
    num_models = len(models)

    # Initialize the velocity of each model
    max_V = torch.tensor(0.1).cuda
    current_V = {}
    current_position = {}
    best_position = {}
    current_acc = []
    best_acc = []
    best_loss = []
    best_global = 0
    Wmax = Wmax
    Wmin = Wmin
    c1 = c1
    c2 = c2

    c = list(models[0].state_dict().keys())
    train_acc_history = []
    test_acc_history = []

    for i in range(num_models):
        models[i].apply(weights_init)
        for key in c:
            if key.endswith('weight'):
                current_position['%s' %i + key] = models[i].state_dict()[key]
                best_position['%s' %i +key] = copy.deepcopy(models[i].state_dict()[key])
                current_V['%s' %i+key] = torch.randn(models[i].state_dict()[key].shape).cuda()
            elif key.endswith('bias'):
                current_position['%s' %i + key] = models[i].state_dict()[key]
                best_position['%s' %i + key]= copy.deepcopy(models[i].state_dict()[key])
                current_V['%s' %i + key] = torch.randn(models[i].state_dict()[key].shape).cuda()
        models[i].eval()
        current_acc.append(test(models[i], test_loader))
        best_acc.append(test(models[i], test_loader))
        best_global = np.argmax(current_acc)
        best_loss.append(float('inf'))
    print("Initial acc: ", current_acc)

    time_start= time.time()
    for epoch in range(epochs):
        w = Wmax - epoch*(Wmax-Wmin)/epochs
        time_step_start = time.time()
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            for i in range(num_models):
                for key in c:
                    if key.endswith('weight'):
                        current_V['%s' %i + key] - w*current_V['%s' %i + key] + c1 
                        + torch.rand(1).cuda()*(
                                    best_position['%s' % i + key] - current_position[
                                '%s' % i + key]) + c2 * torch.rand(1).cuda()*(
                                                                best_position['%s' % best_global + key] -
                                                                current_position['%s' % i + key])
                        current_V['%s' %i + key][current_V['%s' %i + key] > max_V] = max_V
                        current_V['%s' %i + key][current_V['%s' %i + key] < -max_V] = -max_V
                        b = current_position['%s' %i + key] + current_V['%s' %i + key]
                        current_position['%s' %i + key].copy_(b)
                    elif key.endswith('bias'):
                        current_V['%s' %i + key] = w*current_V['%s' %i + key] + c1*torch.rand(1).cuda()*best_position['%s' %i + key] - current_position['%s' %i + key] + c2*torch.rand(1).cuda()*best_position['%s' %best_global + key] - current_position['%s' %i + key]
                        current_V['%s' %i + key][current_V['%s' %i + key] > max_V] = max_V
                        current_V['%s' %i + key][current_V['%s' %i + key] < -max_V] = -max_V
                        b = current_position['%s' %i + key] + current_V['%s' %i + key]
                        current_position['%s' %i + key].copy_(b)
            
            

                models[i].eval()
                output = models[i](batch_x)
                _, predicted = torch.max(output.data, 1)
                total = batch_y.size(0)
                correct = (predicted == batch_y).sum().item()
                acc = 100 * correct / total
                current_acc[i] = acc
                loss = criterion(output, batch_y).data.cpu().numpy()
                if acc >= best_acc[i]:
                    best_acc[i] = acc
                    best_loss[i] = loss
                    state = models[i].state_dict()
                    if not os.path.isdir('checkpoint_pso'):
                        os.mkdir('checkpoint_pso')
                    torch.save(state, './checkpoint_pso/%s_ckpt.pth' % (i))
                    for key in c:
                        if key.endswith('weight'):
                            best_position['%s' %i + key] = copy.deepcopy(models[i].state_dict()[key])
                        elif key.endswith('bias'):
                            best_position['%s' %i + key] = copy.deepcopy(models[i].state_dict()[key])

            best_global = np.argmax(best_acc)
            if step % 50 == (50-1):
                time_step_end = time.time()
                print('[%d, %5d] acc:' %
                    (epoch+1, step+1), best_acc[best_global], 'time: %.2f' % (time_step_end-time_step_start))
                print('[%d, %5d] loss: %.4f' %
                    (epoch+1, step+1, best_loss[best_global], 'time: %.2f' % (time_step_end-time_step_start)))
                
                time_step_start = time_step_end

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net = VGG('VGG11')
        net.load_state_dict(torch.load('./checkpoint_pso/%s_ckpt.pth' % best_global)[0])
        net = net.to(device)
        net.eval()
        train_acc = test(net, train_loader)
        train_acc_history.append(train_acc)
        test_acc = test(net, test_loader)
        test_acc_history.append(test_acc)
        print('epoch(%d)[train_acc, test_acc]:'%epoch, train_acc, test_acc)

    time_end = time.time()
    print('best_globel:', best_global)
    print('best loss:', best_loss)
    print('best acc:', best_acc)
    print('total time: %.2f' % (time_end-time_start))
    return {
        'train_acc_history': train_acc_history,
        'test_acc_history': test_acc_history,
    }
    # state = state = models[best_globel].state_dict(),  # 将当前的参数视为最好状态保存
    # torch.save(state, './checkpoint_pso/best_ckpt.pth')

if __name__ == "__main__":
    # net = VGG('VGG11')
    net = VGG('VGG11')
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    # with SummaryWriter(comment='VGG-11') as w:
    #     w.add_graph(net, (x,))
    # print(net.state_dict()['features.0.weight'].shape, net.state_dict()['features.0.bias'].shape)
    # c = list(net.state_dict().keys())
    # a = {}
    # d = {}
    # a['features.0.weight'] = net.state_dict()['features.0.weight']
    # d['features.0.weight'] = net.state_dict()['features.0.weight']
    # print(d['features.0.weight'])
    # b = torch.zeros(net.state_dict()['features.0.weight'].shape)
    # a['features.0.weight'].copy_(b)
    # print(d['features.0.weight'])
    print(net.state_dict())

        
