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
		this.inputDim[0] = 1;
		this.inputDim[1] = 1;
		this.inputDim[2] = inputLength;
		this.outputDim[0] = 1;
		this.outputDim[1] = 1;
		this.outputDim[2] = outputLength;
		
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
	public void feedForward(double[][][] inputActivations) {		
		//save activations for backpropagation
		for(int p=0; p<this.inputDim[0]; p++) {
			for(int i=0; i<this.inputDim[1]; i++) {
				for(int j=0; j<this.inputDim[2]; j++) {
					this.inActivations[p*this.inputDim[1]*this.inputDim[2] + i*this.inputDim[2] + j] = inputActivations[p][i][j];
				}
			}
		}
		
		
		//array of output activations to return
		double[][][] outActivations = new double[this.outputDim[0]][this.outputDim[1]][this.outputDim[2]];
		
		
		//calculate z and a
		//for each neuron in output layer
		for(int j=0; j<this.outputDim[2]; j++) {	
			double tmpZ = 0;
			for(int k=0; k<this.inputDim[2]; k++) {	//for each neuron in input layer
				tmpZ = tmpZ + this.inActivations[k]*this.weights[j][k];		//add weighted output
			}
			tmpZ = tmpZ + this.outBiases[j];				//add bias
			this.outZs[j] = tmpZ;						//store z value
			outActivations[0][0][j] = actFn(this.outZs[j]);				//store a value
		}
		
		//continue feeding forward if we can, unless the next layer is the final layer
		if(this.getNextLayer() != null) {
			this.getNextLayer().feedForward(outActivations);
		}
	}
	
	//given the errors for its output layer,
	//	returns the errors for its input layer
	//	increments weight/bias gradients.
	//does not actually update weights.
	public void backpropagate(double[][][] outputErrors) {
		//save output delta
		for(int q=0; q<this.outputDim[0]; q++) {
			for(int m=0; m<this.outputDim[1]; m++) {
				for(int n=0; n<this.outputDim[2]; n++) {
					this.outDeltas[q*this.outputDim[1]*this.outputDim[2] + m*this.outputDim[2] + n] = outputErrors[q][m][n];
				}
			}
		}
		
		//error for the input layer to pass on
		double[][][] inDeltas = new double[this.inputDim[0]][this.inputDim[1]][this.inputDim[2]];
		
		//calculate errors
		for(int k=0; k<this.inputDim[2]; k++) {
			for(int j=0; j<this.outputDim[2]; j++) {	//sum over j
				inDeltas[0][0][k] = inDeltas[0][0][k] + this.outDeltas[j]*this.weights[j][k];
			}
			inDeltas[0][0][k] = inDeltas[0][0][k]*this.inActivations[k]*(1-this.inActivations[k]);	//multiply by sigma prime
		}
		
		//increment bias gradients for the output layer
		for(int j=0; j<this.outputDim[2]; j++) {
			this.outBiasesGrad[j] = this.outBiasesGrad[j] + this.outDeltas[j];
		}
		
		//increment weight gradients
		for(int j=0; j<this.outputDim[2]; j++) {
			for(int k=0; k<this.inputDim[2]; k++) {
				this.weightsGrad[j][k] = this.weightsGrad[j][k] + this.outDeltas[j]*this.inActivations[k];
			}
		}
		
		//continue backpropagating
		if(this.getPrevLayer() != null) {
			this.getPrevLayer().backpropagate(inDeltas);
		}
	}
	
	//updates weights and biases from saved gradients
	//resets gradients to zero
	public void updateWeights(double learningRate, int miniBatchSize) {
		//update biases and reset bias gradients
		for(int j=0; j<this.outputDim[2]; j++) {
			this.outBiases[j] = this.outBiases[j] - learningRate*this.outBiasesGrad[j]/miniBatchSize;
			this.outBiasesGrad[j] = 0;
		}
		
		//update weights and reset weight gradients
		for(int j=0; j<this.outputDim[2]; j++) {
			for(int k=0; k<this.inputDim[2]; k++) {
				this.weights[j][k] = this.weights[j][k] - learningRate*this.weightsGrad[j][k]/miniBatchSize;
				this.weightsGrad[j][k] = 0;
			}
		}
	}
	
	//sigmoid activation function
	private double actFn(double z) {
		return 1.0/(1.0+Math.pow(Math.E, -z));
	}
}
