# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
class BatchNorm1d(nn.Module):
	# TODO START
	def __init__(self, num_features, momentum, eps):
		super(BatchNorm1d, self).__init__()
		self.num_features = num_features # hidden layer neutral number

		# Parameters
		self.weight = Parameter(torch.ones(num_features))
		self.bias = Parameter(torch.zeros(num_features))

		# Store the average mean and variance
		self.register_buffer('running_mean', torch.zeros(num_features))
		self.register_buffer('running_var', torch.ones(num_features))
		
		# Initialize your parameter
		self.momentum = momentum
		self.eps = eps

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		if self.training:
			batch_mean = torch.mean(input, dim=0)
			batch_var = torch.var(input, dim=0)
			# exponential moving average
			self.running_mean = (1 - self.momentum) * self.running_mean.data + self.momentum * batch_mean
			self.running_var = (1 - self.momentum) * self.running_var.data + self.momentum * batch_var
		else:
			batch_mean = self.running_mean.data
			batch_var = self.running_var.data
		
		normalized_output = (input - batch_mean) / torch.sqrt(batch_var + self.eps)
		normalized_output = self.weight * normalized_output + self.bias

		return normalized_output
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
		if self.training:
			output = torch.bernoulli(torch.ones_like(input), (1 - self.p)) * input
			output /= (1 - self.p)
		else:
			output = input
			
		return output
	# TODO END

class Model(nn.Module):
	def __init__(self, drop_rate=0.5):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here
		self.layers = nn.Sequential(
			nn.Linear(3 * 32 * 32, 512),
			BatchNorm1d(512, 0.1, 1e-5),
			nn.ReLU(),
			Dropout(drop_rate),
			nn.Linear(512, 10)
		)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):
		# TODO START
		# the 10-class prediction output is named as "logits"
		logits = self.layers(x)
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc
