from network import Network
from utils import LOG_INFO
from layers import Selu, Swish, Linear, Gelu, Relu, Sigmoid
from loss import MSELoss, SoftmaxCrossEntropyLoss, HingeLoss, FocalLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
from torch.utils.tensorboard import SummaryWriter


train_data, test_data, train_label, test_label = load_mnist_2d('data')

writer = SummaryWriter('log')
model = Network()
loss = None

def zero_hidden_layer_test():
    default_config = {
        'learning_rate': 0.01,
        'weight_decay': 0,
        'momentum': 0.9,
        'batch_size': 64,
        'max_epoch': 100,
        'disp_freq': 50,
        'test_epoch': 1
    }
    
    for loss in [MSELoss(name='MSELoss'), SoftmaxCrossEntropyLoss(name='SoftmaxCrossEntropyLoss'), HingeLoss(name='HingeLoss', margin=0.03), FocalLoss('FocalLoss', alpha=0.9)]:
        # use default config
        model = Network()
        model.add(Linear('fc1', 784, 10, 0.01))
        model.add(Sigmoid('Sigmoid'))
        print(f'TRAIN with zero hidden layer with {loss.name}')
        for epoch in range(default_config['max_epoch']):
            return_train_data = train_net(model, loss, default_config, train_data, train_label, default_config['batch_size'], default_config['disp_freq'])
            writer.add_scalars(f'train_loss', {loss.name: return_train_data[0]}, epoch)
            writer.add_scalars(f'train_acc', {loss.name: return_train_data[1]}, epoch)

            return_test_data = test_net(model, loss, test_data, test_label, default_config['batch_size'])
            writer.add_scalars(f'test_loss', {loss.name: return_test_data[0]}, epoch)
            writer.add_scalars(f'test_acc', {loss.name: return_test_data[1]}, epoch)



def one_hidden_layer_loss_test():
    default_config = {
        'learning_rate': 0.1,
        'weight_decay': 0,
        'momentum': 0.9,
        'batch_size': 64,
        'max_epoch': 100,
        'disp_freq': 50,
        'test_epoch': 1
    }
    
    for loss in [MSELoss(name='MSELoss'), SoftmaxCrossEntropyLoss(name='SoftmaxCrossEntropyLoss'), HingeLoss(name='HingeLoss', margin=0.03), FocalLoss('FocalLoss', alpha=0.9)]:
        # use default config
        model = Network()
        model.add(Linear('fc1', 784, 100, 0.01))
        model.add(Sigmoid('Sigmoid'))
        model.add(Linear('fc2', 100, 10, 0.01))
        model.add(Sigmoid('Sigmoid'))
        print(f'TRAIN with one hidden layer with {loss.name}')
        for epoch in range(default_config['max_epoch']):
            return_train_data = train_net(model, loss, default_config, train_data, train_label, default_config['batch_size'], default_config['disp_freq'])
            writer.add_scalars(f'train_loss', {loss.name: return_train_data[0]}, epoch)
            writer.add_scalars(f'train_acc', {loss.name: return_train_data[1]}, epoch)

            return_test_data = test_net(model, loss, test_data, test_label, default_config['batch_size'])
            writer.add_scalars(f'test_loss', {loss.name: return_test_data[0]}, epoch)
            writer.add_scalars(f'test_acc', {loss.name: return_test_data[1]}, epoch)

def one_hidden_layer_activate_test():
    default_config = {
        'learning_rate': 0.01,
        'weight_decay': 0,
        'momentum': 0.9,
        'batch_size': 64,
        'max_epoch': 50,
        'disp_freq': 50,
        'test_epoch': 1
    }
    
    for activate in [Sigmoid('Sigmoid'), Selu('Selu'), Swish('Swish'), Gelu('Gelu'), Relu('Relu')]:
        # use default config
        model = Network()
        model.add(Linear('fc1', 784, 100, 0.01))
        model.add(activate)
        model.add(Linear('fc2', 100, 10, 0.01))
        model.add(Sigmoid('Sigmoid'))
        loss = HingeLoss(name='Hinge', margin=0.03)
        print(f'TRAIN with one hidden layer with {activate.name}')
        for epoch in range(default_config['max_epoch']):
            return_train_data = train_net(model, loss, default_config, train_data, train_label, default_config['batch_size'], default_config['disp_freq'])
            writer.add_scalars(f'train_loss', {activate.name: return_train_data[0]}, epoch)
            writer.add_scalars(f'train_acc', {activate.name: return_train_data[1]}, epoch)

            return_test_data = test_net(model, loss, test_data, test_label, default_config['batch_size'])
            writer.add_scalars(f'test_loss', {activate.name: return_test_data[0]}, epoch)
            writer.add_scalars(f'test_acc', {activate.name: return_test_data[1]}, epoch)

def two_hidden_layer_loss_test():
    default_config = {
        'learning_rate': 0.01,
        'weight_decay': 0,
        'momentum': 0.9,
        'batch_size': 64,
        'max_epoch': 50,
        'disp_freq': 50,
        'test_epoch': 1
    }
    
    for loss in [HingeLoss(name='HingeLoss', margin=0.03), FocalLoss('FocalLoss', alpha=0.9)]:
        # use default config
        if loss.name == 'HingeLoss':
            default_config['learning_rate'] = 0.01
        else:
            default_config['learning_rate'] = 0.1
        model = Network()
        model.add(Linear('fc1', 784, 256, 0.01))
        model.add(Swish('Swish'))
        model.add(Linear('fc2', 256, 100, 0.01))
        model.add(Swish('Swish'))
        model.add(Linear('fc3', 100, 10, 0.01))
        model.add(Sigmoid('Sigmoid'))
        print(f'TRAIN with two hidden layers with {loss.name}')

        for epoch in range(default_config['max_epoch']):
            return_train_data = train_net(model, loss, default_config, train_data, train_label, default_config['batch_size'], default_config['disp_freq'])
            writer.add_scalars(f'train_loss', {loss.name: return_train_data[0]}, epoch)
            writer.add_scalars(f'train_acc', {loss.name: return_train_data[1]}, epoch)

            return_test_data = test_net(model, loss, test_data, test_label, default_config['batch_size'])
            writer.add_scalars(f'test_loss', {loss.name: return_test_data[0]}, epoch)
            writer.add_scalars(f'test_acc', {loss.name: return_test_data[1]}, epoch)

def two_hidden_layer_activate_test():
    default_config = {
        'learning_rate': 0.01,
        'weight_decay': 0,
        'momentum': 0.9,
        'batch_size': 64,
        'max_epoch': 50,
        'disp_freq': 50,
        'test_epoch': 1
    }
    
    for activate1 in [Selu('Selu'), Swish('Swish'), Gelu('Gelu')]:
        # use default config
        for activate2 in [Selu('Selu'), Swish('Swish'), Gelu('Gelu')]:
            model = Network()
            model.add(Linear('fc1', 784, 256, 0.01))
            model.add(activate1)
            model.add(Linear('fc2', 256, 100, 0.01))
            model.add(activate2)
            model.add(Linear('fc3', 100, 10, 0.01))
            model.add(Sigmoid('Sigmoid'))
            loss = HingeLoss(name='Hinge', margin=0.03)
            print(f'TRAIN with teo hidden layer with {activate1.name}_{activate2.name}')
            for epoch in range(default_config['max_epoch']):
                return_train_data = train_net(model, loss, default_config, train_data, train_label, default_config['batch_size'], default_config['disp_freq'])
                writer.add_scalars(f'train_loss', {f'{activate1.name}_{activate2.name}': return_train_data[0]}, epoch)
                writer.add_scalars(f'train_acc', {f'{activate1.name}_{activate2.name}': return_train_data[1]}, epoch)

                return_test_data = test_net(model, loss, test_data, test_label, default_config['batch_size'])
                writer.add_scalars(f'test_loss', {f'{activate1.name}_{activate2.name}': return_test_data[0]}, epoch)
                writer.add_scalars(f'test_acc', {f'{activate1.name}_{activate2.name}': return_test_data[1]}, epoch)


def focal_test():
    default_config = {
        'learning_rate': 0.1,
        'weight_decay': 0,
        'momentum': 0.9,
        'batch_size': 64,
        'max_epoch': 50,
        'disp_freq': 50,
        'test_epoch': 1
    }
    
    for loss in [SoftmaxCrossEntropyLoss(name='SoftmaxCrossEntropyLoss'), FocalLoss('FocalLoss', alpha=0.9)]:
        # use default config
        model = Network()
        model.add(Linear('fc1', 784, 100, 0.01))
        model.add(Gelu('Gelu'))
        model.add(Linear('fc2', 100, 10, 0.01))
        model.add(Sigmoid('Sigmoid'))
        print(f'TRAIN with one hidden layer with {loss.name}')
        for epoch in range(default_config['max_epoch']):
            return_train_data = train_net(model, loss, default_config, train_data, train_label, default_config['batch_size'], default_config['disp_freq'])
            writer.add_scalars(f'train_loss', {loss.name: return_train_data[0]}, epoch)
            writer.add_scalars(f'train_acc', {loss.name: return_train_data[1]}, epoch)

            return_test_data = test_net(model, loss, test_data, test_label, default_config['batch_size'])
            writer.add_scalars(f'test_loss', {loss.name: return_test_data[0]}, epoch)
            writer.add_scalars(f'test_acc', {loss.name: return_test_data[1]}, epoch)

def overfit_test():
    default_config = {
        'learning_rate': 0.01,
        'weight_decay': 0,
        'momentum': 0,
        'batch_size': 64,
        'max_epoch': 50,
        'disp_freq': 50,
        'test_epoch': 1
    }
    
    for weight_decay in [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        # use default config
        default_config['weight_decay'] = weight_decay
        model = Network()
        model.add(Linear('fc1', 784, 100, 0.01))
        model.add(Gelu('Sigmoid'))
        model.add(Linear('fc2', 100, 10, 0.01))
        model.add(Sigmoid('Sigmoid'))
        loss = MSELoss(name='loss')
        print(f'TRAIN with one hidden layer with {weight_decay}')
        for epoch in range(default_config['max_epoch']):
            return_train_data = train_net(model, loss, default_config, train_data, train_label, default_config['batch_size'], default_config['disp_freq'])
            writer.add_scalars(f'train_loss', {f'weight_decay_{weight_decay}': return_train_data[0]}, epoch)
            writer.add_scalars(f'train_acc', {f'weight_decay_{weight_decay}': return_train_data[1]}, epoch)

            return_test_data = test_net(model, loss, test_data, test_label, default_config['batch_size'])
            writer.add_scalars(f'test_loss', {f'weight_decay_{weight_decay}': return_test_data[0]}, epoch)
            writer.add_scalars(f'test_acc', {f'weight_decay_{weight_decay}': return_test_data[1]}, epoch)

if __name__ == '__main__':
    zero_hidden_layer_test()
    one_hidden_layer_loss_test()
    one_hidden_layer_activate_test()
    two_hidden_layer_loss_test()
    two_hidden_layer_activate_test()
    focal_test()
    overfit_test()