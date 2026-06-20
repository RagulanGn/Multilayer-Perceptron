import numpy as np

class Value:
	def __init__(self, number, childrens = None):
		self.number = np.array(number, dtype=float)
		self.childrens = childrens or []
		self._backward = lambda: None
		self._grad = np.zeros_like(self.number, dtype=float)
	
	def __repr__(self):
		return (f"Value : \n{self.number} \n grad : {self._grad}")
	
	def backward(self):
		topo = []
		visited = set()
		def build_topo(v):
			if v not in visited :
				visited.add(v)
				for child in v.childrens:
					build_topo(child)
				topo.append(v)
		build_topo(self)

		self._grad = np.ones_like(self.number)

		for node in reversed(topo):
			node._backward()

	#################################################################################################
	#-------------------------------------------OPERATION-------------------------------------------#
	#################################################################################################

	#Broadcast rules : Dimension equals, one of them is one
	#Right side alignment 

	#Unbroadcast rules : The gradient should have the same shape as the corresponding matrix
	# If it has not its means broadcast happened
	# Squash the left side value into one until gradient and matrix have the same dim except those who equals ones 
	# Check the remaining dimension some of them may need to be squashed to one number
	# Add every dimension that should not exist 

	def _unbroadcast(self, grad, shape):
		res = grad
		while (res.ndim > len(shape)):
			res = res.sum(axis=0)

		for axis, dim in enumerate(shape):
			if dim == 1:
				res = res.sum(axis=axis, keepdims=True)
		return res

	def __add__(self, right_operand):
		right_operand = right_operand if isinstance(right_operand, Value) else Value(right_operand)
		add = Value(self.number + right_operand.number)
		add.childrens = [self, right_operand]
		def _backward():
			# self._grad += 1 * add._grad
			# right_operand._grad += 1 * add._grad
			self._grad += self._unbroadcast(add._grad, self.number.shape)
			right_operand._grad += self._unbroadcast(add._grad, right_operand.number.shape)
		add._backward = _backward
		return add

	def __mul__(self, right_operand):
		right_operand = right_operand if isinstance(right_operand, Value) else Value(right_operand)
		mul = Value(self.number * right_operand.number)
		mul.childrens = [self, right_operand]
		def _backward():
			self._grad += self._unbroadcast(right_operand.number * mul._grad, self.number.shape)
			right_operand._grad += self._unbroadcast(self.number * mul._grad, right_operand.number.shape)
		mul._backward = _backward
		return mul

	def __pow__(self, right_operand):
		power = Value(self.number ** right_operand)
		power.childrens = [self]
		def _backward():
			self._grad += right_operand * (self.number ** (right_operand - 1)) * power._grad
		power._backward = _backward
		return power

	def __matmul__(self, right_operand):
		right_operand = right_operand if isinstance(right_operand, Value) else Value(right_operand)
		matmul = Value(self.number @ right_operand.number)
		matmul.childrens = [self, right_operand]
		def _backward():
			self._grad += matmul._grad @ right_operand.number.T
			right_operand._grad += self.number.T @ matmul._grad
		matmul._backward = _backward
		return matmul

	def __neg__(self):
		return self * -1

	def __sub__(self, right_operand):
		return self + (-right_operand)

	def __truediv__(self, right_operand):
		right_operand = right_operand if isinstance(right_operand, Value) else Value(right_operand)
		return self * (right_operand ** -1)

	def __rmul__(self, right_operand):
		return self * right_operand

	def __radd__(self, right_operand):
		return self + right_operand

	def __rsub__(self, right_operand):
		return Value(right_operand) - self

	def __rtruediv__(self, right_operand):
		return Value(right_operand) * (self ** -1)

	@property
	def T(self):
		transpose = Value(self.number.T, childrens=[self])
		def _backward():
			self._grad += transpose._grad.T
		transpose._backward = _backward
		return transpose

	#################################################################################################
	#-------------------------------------------FUNCTION--------------------------------------------#
	#################################################################################################

	def ReLU(self):
		relu = Value(np.maximum(0, self.number))
		relu.childrens = [self]
		def _backward():
			self._grad += (self.number > 0) * relu._grad
		relu._backward = _backward
		return relu

	def exp(self):
		exp = Value(np.exp(self.number))
		exp.childrens = [self]
		def _backward():
			self._grad += exp.number * exp._grad
		exp._backward = _backward
		return exp

	def log(self):
		eps = 1e-15
		safe = np.clip(self.number, eps, None)

		log = Value(np.log(safe))
		log.childrens = [self]
		def _backward():
			self._grad += (1.0 / (safe)) * log._grad
		log._backward = _backward
		return log

	def sum(self, axis=None, keepdims=False):
		s = Value(self.number.sum(axis=axis, keepdims=keepdims))
		s.childrens = [self]
		def _backward():
			grad_reshaped = s._grad
			if axis is not None and not keepdims:
				actual_axis = axis if isinstance(axis, (list, tuple)) else (axis, )
				actual_shape = list(self.number.shape)
				for ax in actual_axis:
					actual_shape[ax] = 1
				grad_reshaped = s._grad.reshape(actual_shape)
			self._grad += np.ones_like(self.number) * grad_reshaped

		s._backward = _backward
		return s

	def mean(self, axis=None, keepdims=False):
		m = Value(self.number.mean(axis=axis, keepdims=False))
		m.childrens = [self]

		if axis is None:
			N = self.number.size
		else:
			actual_axis = axis if isinstance(axis, (list, tuple)) else (axis,)
			N = 1
			for ax in actual_axis:
				N *= self.number.shape[ax]
		def _backward():
			grad_reshaped = m._grad
			if axis is not None and not keepdims:
				actual_axis = axis if isinstance(axis, (list, tuple)) else (axis, )
				actual_shape = list(self.number.shape)
				for ax in actual_axis:
					actual_shape[ax] = 1
				grad_reshaped = m._grad.reshape(actual_shape)
			self._grad += (np.ones_like(self.number) * grad_reshaped) / N
		m._backward = _backward
		return m
