from __future__ import division
import numpy as np


class MSELoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        '''Your codes here'''
        return (np.sum((input - target) ** 2)) / input.shape[0]
        pass
        # TODO END

    def backward(self, input, target):
		# TODO START
        '''Your codes here'''
        return 2 * (input - target) / input.shape[0]
        pass
		# TODO END


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name
        self.softmax_output = None

    def forward(self, input, target):
        # TODO START
        '''Your codes here'''
        exp_input = np.exp(input)
        sum_exp = np.sum(exp_input, axis=-1, keepdims=True)
        self.softmax_output = exp_input / sum_exp
        loss = np.sum(-np.log(self.softmax_output) * target, axis=-1).mean()
        return loss
        pass
        # TODO END

    def backward(self, input, target):
        # TODO START
        '''Your codes here'''
        return (self.softmax_output - target) / input.shape[0]
        pass
        # TODO END


class HingeLoss(object):
    def __init__(self, name, margin=5):
        self.name = name
        self.margin = margin

    def forward(self, input, target):
        # TODO START 
        '''Your codes here'''
        margin_input = np.maximum(0, self.margin - (input * target) + input)
        loss = (margin_input.sum(axis=-1) - self.margin).mean()
        return loss
        pass
        # TODO END

    def backward(self, input, target):
        # TODO START
        '''Your codes here'''
        grad = ((self.margin - (input * target) + input) > 0).astype('float')
        grad -= target * grad.sum(axis=-1, keepdims=True)
        return grad / input.shape[0]
        pass
        # TODO END

# Bonus
class FocalLoss(object):
    def __init__(self, name, alpha=None, gamma=2.0):
        self.name = name
        if alpha is None:
            self.alpha = [0.1 for _ in range(10)]
        else:
            self.alpha = [alpha for _ in range(10)]
        self.gamma = gamma
        self.softmax_output = None

    def forward(self, input, target):
        # TODO START
        '''Your codes here'''
        exp_input = np.exp(input)
        sum_exp = np.sum(exp_input, axis=-1, keepdims=True)
        self.softmax_output = exp_input / sum_exp
        alpha = np.array(self.alpha)
        alpha_weight = alpha * target + (1 - alpha) * (1 - target)
        gamma_weight = (1 - self.softmax_output) ** self.gamma
        loss = np.sum(-np.log(self.softmax_output) * target * alpha_weight * gamma_weight, axis=-1).mean()
        return loss
        pass
        # TODO END
       

    def backward(self, input, target):
        # TODO START
        '''Your codes here'''
        # Reference: https://zhuanlan.zhihu.com/p/32631517
        # Reference: 与薛晗同学（2021010725）讨论实现方法以及alpha参数的选择
        alpha = np.array(self.alpha)
        grad = np.zeros_like(input)
        gamma_grad = ((1 - self.softmax_output) ** (self.gamma - 1)) * (1 - self.softmax_output - self.gamma * self.softmax_output * np.log(self.softmax_output))
        for i in range(input.shape[0]):
            # i: batch num
            t = np.where(target[i] == 1)
            grad[i] = np.where(target[i] == 1, (gamma_grad * (self.softmax_output - 1))[i][t], (gamma_grad[i][t] * self.softmax_output)[i])

        grad *= (alpha[0] / input.shape[0])

        return grad
        pass
        # TODO END
        
