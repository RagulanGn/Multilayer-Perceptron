import numpy as np

class NesterovMomentum():
	def __init__(self, lr=10e-3, b1=0.9):
		self.lr = lr
		self.b1 = b1
		self.m = {} #Dictionnary of all momentum for each p (parameters)

	def __call__(self, parameters):

		for p in parameters:
			if p not in self.m:
				self.m[p] = np.zeros_like(p.number)
			grad = p._grad
			self.m[p] = self.b1 * self.m[p] + grad
			p.number -= self.lr * (self.b1 * self.m[p] + grad)

class Adam():
	def __init__(self, lr=10e-3, b1=0.9, b2=0.99):	
		self.lr = lr	#Learning rate
		self.b1 = b1	#ratio of Momentum according to previous gradient
		self.b2 = b2	#ratio of Second Momentum according to variance of the gradient
		self.ep = 1e-8	#Constant use to not divide by zero

		self.t = 0
		self.m = {}
		self.v = {}

	def __call__(self, parameters) :
		self.t += 1
		for p in parameters:
			if p not in self.m:
				self.m[p] = np.zeros_like(p.number)
				self.v[p] = np.zeros_like(p.number)
			grad = p._grad
			self.m[p] = self.b1 * self.m[p] + (1 - self.b1) * grad
			self.v[p] = self.b2 * self.v[p] + (1 - self.b2) * grad ** 2

			m_hat = self.m[p] / (1 - self.b1 ** self.t)
			v_hat = self.v[p] / (1 - self.b2 ** self.t)
			p.number -= self.lr * m_hat / (np.sqrt(v_hat) + self.ep)
