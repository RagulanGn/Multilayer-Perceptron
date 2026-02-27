Notions

feedforward: La communication entre les neuronnes ne se fait que d un sens, on ne peut qu avancer
backpropagation: Cependant l erreur se propage elle dans le sens inverse
gradient descent: Algorithm that helps find the minimum of a function
using small step along the gradient (those step should lead to the minimum)

Structure

Model :
	Layer : 
		Input layer :
			No transformation on input (meaning no matrix applied)
		Hidden layer :
			Neuron :
				- Activation function
				- Weight matrix
			Biais neuron :
				- Add biais
		Output layer :
			Final output (could be matricial ???)
		!!! In our program Output layer is the last "hidden layer" !!!
		Layer have an input matrix size and an output matrix size

What we need to add ?

	- Loss Calculation (After each epoch)
	- Backpropagation

Practical exception :
	In theory we could use a different activation for each neuron but in practice we use the same function 
for each neuron of a layer. That allows us to use the function on the matrix directly and save a good amount
of calculation. And we dont sacrifice too much on the precision of the model because of the nature of MLP.

Gradient Calculation :

a = activation_function(z) = z | 0
zout = W2 @ Input + B

dLoss/dW2 = dLoss/da2 * da2/dzout * dzout/dw2
dLoss/dW1 = dLoss/da2 * da2/dzout * dzout/da1 * da1/dzhidden * dzhidden/dW1

dLoss/dW2 = delta2 * dzout/dw2 = delta2 * a1 (input returned by activation funciton of layer 1)
	with delta2 = dLoss/da2 * da2/dzout
dLoss/dW1 = delta1 * dzhidden/dW1 = delta1 * input
	with delta1 = delta2 * dzout/da1 * da1/dzhidden


Mathematical function:

	Activation function : Add non linearity to the algorithm can take multiple form
	Weight Sum : Z = AW + B -> What we try to correct

	Jacobian - > Should use jacobian for derivation of activation function softmax
		Problem should replace the multiplication of activation function by a matrix multiplication
		If we separate output layer and input layer it should be good ?!