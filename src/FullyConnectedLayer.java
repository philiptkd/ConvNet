import java.util.Random;

/*
 * Each neuron in this layer will be connected to all the neurons in the previous volume.
 * This layer creates and maintains the nodes of its output layer.
 * It only keeps track of the activations of its input layer.
 * Uses a sigmoid as its activation function.
 */
public class FullyConnectedLayer extends Layer{
	public double[][] weights;
	public double[] inActivations;
	public double[] outBiases;
	private double[] outZs;
	public double[] outDeltas;
	private double[][] weightsGrad;
	private double[] outBiasesGrad;
	private static Random rand = new Random();
	
	//constructor
	public FullyConnectedLayer(int inputLength, int outputLength) {
		//save lengths of input and output
		this.inputLength = inputLength;
		this.outputLength = outputLength;
		
		//create array to hold input activations
		this.inActivations = new double[inputLength];
		
		//create output layer
		this.outZs = new double[outputLength];
		this.outDeltas = new double[outputLength];
		
		//create weights and biases (initially zeros)
		this.weights = new double[outputLength][inputLength];
		this.weightsGrad = new double[outputLength][inputLength];
		this.outBiases = new double[outputLength];
		this.outBiasesGrad = new double[outputLength];
		
		//initialize weights to have an acceptable variance
		for(int k=0; k<inputLength; k++) {
			for(int j=0; j<outputLength; j++) {
				this.weights[j][k] = rand.nextGaussian()/Math.sqrt(inputLength);	//std dev. = 1/numInputNodes
			}
		}
		
		//how biases are initialized isn't very important		
		for(int i=0; i<outBiases.length; i++) {
			this.outBiases[i] = (rand.nextDouble()- 0.5)*2;
		}
	}
	
	//given the activations for its input layer,
	//	returns activations of its output layer
	@Override
	public void feedForward(double[] inputActivations) {		
		//save activations for backpropagation
		for(int j=0; j<this.inputLength; j++) {
			this.inActivations[j] = inputActivations[j];
		}
		
		//array of output activations to return
		double[] outActivations = new double[this.outputLength];
		
		//calculate z and a
		for(int j=0; j<this.outputLength; j++) {	//for each neuron in output layer
			double tmpZ = 0;
			for(int k=0; k<this.inputLength; k++) {	//for each neuron in input layer
				tmpZ = tmpZ + this.inActivations[k]*this.weights[j][k];		//add weighted output
			}
			tmpZ = tmpZ + this.outBiases[j];				//add bias
			this.outZs[j] = tmpZ;						//store z value
			outActivations[j] = actFn(this.outZs[j]);				//store a value
		}
		
		//continue feeding forward if we can, unless the next layer is the final layer
		if(this.nextLayer != null) {
			this.nextLayer.feedForward(outActivations);
		}
	}
	
	//given the errors for its output layer,
	//	returns the errors for its input layer
	//	increments weight/bias gradients.
	//does not actually update weights.
	public double[] backpropagate(double[] outputErrors) {
		//save output delta
		for(int j=0; j<this.outputLength; j++) {
			this.outDeltas[j] = outputErrors[j];
		}
		
		//error for the input layer to return
		double[] inDeltas = new double[inputLength];
		
		for(int k=0; k<this.inputLength; k++) {
			for(int j=0; j<this.outputLength; j++) {	//sum over j
				inDeltas[k] = inDeltas[k] + this.outDeltas[j]*this.weights[j][k];
			}
			inDeltas[k] = inDeltas[k]*this.inActivations[k]*(1-this.inActivations[k]);	//multiply by sigma prime
		}
		
		//increment bias gradients for the output layer
		for(int j=0; j<this.outputLength; j++) {
			this.outBiasesGrad[j] = this.outBiasesGrad[j] + this.outDeltas[j];
		}
		
		//increment weight gradients
		for(int j=0; j<this.outputLength; j++) {
			for(int k=0; k<this.inputLength; k++) {
				this.weightsGrad[j][k] = this.weightsGrad[j][k] + this.outDeltas[j]*this.inActivations[k];
			}
		}
		
		return inDeltas;
	}
	
	//updates weights and biases from saved gradients
	//resets gradients to zero
	public void updateWeights(double learningRate, int miniBatchSize) {
		//update biases and reset bias gradients
		for(int j=0; j<this.outputLength; j++) {
			outBiases[j] = outBiases[j] - learningRate*outBiasesGrad[j]/miniBatchSize;
			outBiasesGrad[j] = 0;
		}
		
		//update weights and reset weight gradients
		for(int j=0; j<outputLength; j++) {
			for(int k=0; k<inputLength; k++) {
				weights[j][k] = weights[j][k] - learningRate*weightsGrad[j][k]/miniBatchSize;
				weightsGrad[j][k] = 0;
			}
		}
	}
	
	//sigmoid activation function
	private double actFn(double z) {
		return 1.0/(1.0+Math.pow(Math.E, -z));
	}
}
