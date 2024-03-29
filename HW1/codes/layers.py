import numpy as np


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor

class Relu(Layer):
	def __init__(self, name):
		super(Relu, self).__init__(name)

	def forward(self, input):
		self._saved_for_backward(input)
		return np.maximum(0, input)

	def backward(self, grad_output):
		input = self._saved_tensor
		return grad_output * (input > 0)

class Sigmoid(Layer):
	def __init__(self, name):
		super(Sigmoid, self).__init__(name)

	def forward(self, input):
		output = 1 / (1 + np.exp(-input))
		self._saved_for_backward(output)
		return output

	def backward(self, grad_output):
		output = self._saved_tensor
		return grad_output * output * (1 - output)

class Selu(Layer):
    def __init__(self, name):
        super(Selu, self).__init__(name)

    def forward(self, input):
        # TODO START
        '''Your codes here'''
        self._saved_for_backward(input)
        output = 1.0507 * np.where(input > 0, input, 1.67326 * (np.exp(input) - 1))
        return output
        pass
        # TODO END

    def backward(self, grad_output):
        # TODO START
        '''Your codes here'''
        input = self._saved_tensor
        return np.where(input > 0, grad_output * 1.0507, grad_output * 1.0507 * 1.67326 * np.exp(input))
        pass
        # TODO END

class Swish(Layer):
    def __init__(self, name):
        super(Swish, self).__init__(name)

    def forward(self, input):
        # TODO START
        '''Your codes here'''
        output = input / (1 + np.exp(-input))
        self._saved_for_backward(input)
        return output
        pass
        # TODO END

    def backward(self, grad_output):
        # TODO START
        '''Your codes here'''
        input = self._saved_tensor
        sigmoid_input = 1 / (1 + np.exp(-input))
        # return (1 + np.exp(-input) + input * np.exp(-input)) / (1 + np.exp(-input)) ** 2
        return grad_output * (sigmoid_input + input * sigmoid_input * (1 - sigmoid_input))
        pass
        # TODO END

class Gelu(Layer):
    def __init__(self, name):
        super(Gelu, self).__init__(name)

    def forward(self, input):
        # TODO START
        '''Your codes here'''
        self._saved_for_backward(input)
        output = 0.5 * input * (1 + np.tanh(np.sqrt(2 / np.pi) * (input + 0.044715 * input ** 3)))
        return output
        pass
        # TODO END
    
    def backward(self, grad_output):
        # TODO START
        '''Your codes here'''
        # Reference: https://zhuanlan.zhihu.com/p/394465965
        input = self._saved_tensor
        sqrt_2, sqrt_pi = np.sqrt(2), np.sqrt(np.pi)
        return grad_output * 0.5 * (np.tanh((sqrt_2 * (0.044715 * input ** 3 + input)) / sqrt_pi) + 
                ((sqrt_2 * input * (0.134145 * input ** 2 + 1) * ((1 / np.cosh(
                (sqrt_2 * (0.044715 * input ** 3 + input)) / sqrt_pi)) ** 2)) / sqrt_pi + 1))
        pass
        # TODO END

class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        # TODO START
        '''Your codes here'''
        self._saved_for_backward(input)
        return np.matmul(input, self.W) + self.b
        pass
        # TODO END

    def backward(self, grad_output):
        # TODO START
        '''Your codes here'''
        input = self._saved_tensor
        self.grad_W = np.matmul(input.T, grad_output)
        self.grad_b = np.sum(grad_output, axis=0)
        grad_input = np.matmul(grad_output, self.W.T)
        return grad_input
        pass
        # TODO END

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b
