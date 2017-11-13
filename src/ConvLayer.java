import java.util.Random;

/*
 * CONV layer will compute the output of neurons that are connected to local regions in the input, 
 * each computing a dot product between their weights and a small region they are connected to in 
 * the input volume. The depth of the resulting volume is the number of filters/kernels used.
 */

/*
 * Parameters:
 * step size (=1?)
 * number of kernels: P (numFilters)
 * kernel dimensions: K1xK2
 * input dimensions: HxWxD
 * cross-correlation: C_p (i,j) = Sum_m,n I(i+m,j+n)K_p(m,n)
 * 		where 0 <= m <= K1-1 and 0 <= n <= K2-1
 * 		and where C_p is defined only where I and K_p are defined
 * output dimensions: (H-K1+1)x(W-K2+1)xP
 */

public class ConvLayer extends Layer{
	//dimensions of input
	public int inputDepth;
	public int inputHeight;
	public int inputWidth;
	
	//dimensions of kernel
	public int numFilters;
	public int kernelDepth;
	public int kernelHeight;
	public int kernelWidth;
	
	//dimensions of output
	public int outputDepth;
	public int outputHeight;
	public int outputWidth;
	
	//has dimensions of [inputDepth][inputHeight][inputWidth]
	public double[][][] inActivations;
	
	//has dimensions of [numFilters][kernelDepth][kernelHeight][kernelWidth]
	public double[][][][] kernels;
	private double[][][][] kernelGrad;
	
	//has length of numFilters
	public double[] outBiases;
	private double[] outBiasesGrad;
	
	//has dimensions of [outputDepth][outputHeight][outputWidth]
	private double[][][] outZs;
	public double[][][] outDeltas;
	
	private static Random rand = new Random();
	
	//constructor
	public ConvLayer(int[] inputDimensions, int[] kernelDimensions) throws LayerCompatibilityException {
		//check length of parameters
		if(inputDimensions.length != 3) {
			throw new LayerCompatibilityException("ConvLayer inputDimensions should be length 3 but was length "+ inputDimensions.length + ".");
		}
		if(kernelDimensions.length != 4) {
			throw new LayerCompatibilityException("ConvLayer kernelDimensions should be length 4 but was length "+ kernelDimensions.length + ".");
		}
		if(kernelDimensions[1] != inputDimensions[0]) {	//if the depths aren't the same
			throw new LayerCompatibilityException("ConvLayer inputDimensions and kernelDimensions should be given the same depth.");
		}
		
		//save input parameters
		this.inputDepth = inputDimensions[0];
		this.inputHeight = inputDimensions[1];
		this.inputWidth = inputDimensions[2];
		this.numFilters = kernelDimensions[0];
		this.kernelDepth = kernelDimensions[1];
		this.kernelHeight = kernelDimensions[2];
		this.kernelWidth = kernelDimensions[3];
		
		//calculate output dimensions
		this.outputDepth = this.numFilters;
		this.outputHeight = this.inputHeight - this.kernelHeight + 1;	//numRows = (H-k1+1)
		this.outputWidth = this.inputWidth - this.kernelWidth + 1;	//numColumns = (W-k2+1)
		
		//save inputLength and outputLength for compatibility testing purposes
		this.inputLength = this.inputDepth*this.inputHeight*this.inputWidth;
		this.outputLength = this.outputDepth*this.outputHeight*this.outputWidth;
		
		//create arrays to have appropriate sizes
		this.inActivations = new double[this.inputDepth][this.inputHeight][this.inputWidth];
		this.kernels = new double[this.numFilters][this.kernelDepth][this.kernelHeight][this.kernelWidth];
		this.kernelGrad = new double[this.numFilters][this.kernelDepth][this.kernelHeight][this.kernelWidth];
		this.outBiases = new double[this.numFilters];	//one for each filter
		this.outBiasesGrad = new double[this.numFilters];	//one for each filter
		this.outZs = new double[this.outputDepth][this.outputHeight][this.outputWidth];
		this.outDeltas = new double[this.outputDepth][this.outputHeight][this.outputWidth];
		
		//initialize weights to have an acceptable variance
		for(int p=0; p<this.numFilters; p++) {
			for(int d=0; d<this.kernelDepth; d++) {
				for(int k1=0; k1<this.kernelHeight; k1++) {
					for(int k2=0; k2<this.kernelWidth; k2++) {
						this.kernels[p][d][k1][k2] = rand.nextGaussian()/Math.sqrt(this.kernelDepth*this.kernelHeight*this.kernelWidth);
					}
				}
			}
		}
		
		//the initial variance of the biases matters less
		for(int i=0; i<this.outBiases.length; i++) {
			this.outBiases[i] = rand.nextGaussian();
		}
	}
	
	public void feedForward(double[] inputActivations) {
		//save input activations in multidimensional array
		for(int d=0; d<this.inputDepth; d++) {
			for(int h=0; h<this.inputHeight; h++) {
				for(int w=0; w<this.inputWidth; w++) {
					this.inActivations[d][h][w] = inputActivations[d*this.inputWidth*this.inputHeight + h*this.inputWidth + w];
				}
			}
		}
		
		
	}
	
	public void backpropagate(double[] outputErrors) {
		//save output delta in multidimensional array
		for(int d=0; d<this.outputDepth; d++) {
			for(int h=0; h<this.outputHeight; h++) {
				for(int w=0; w<this.outputWidth; w++) {
					this.outDeltas[d][h][w] = outputErrors[d*this.outputWidth*this.outputHeight + h*this.outputWidth + w];
				}
			}
		}
		
		
	}
	
	public void updateWeights(double learningRate, int miniBatchSize) {
		//update biases and reset bias gradients
		for(int j=0; j<this.outBiases.length; j++) {
			this.outBiases[j] = this.outBiases[j] - learningRate*this.outBiasesGrad[j]/miniBatchSize;
			this.outBiasesGrad[j] = 0;
		}
		
		//update weights and reset weight gradients
		for(int p=0; p<this.numFilters; p++) {
			for(int d=0; d<this.kernelDepth; d++) {
				for(int k1=0; k1<this.kernelHeight; k1++) {
					for(int k2=0; k2<this.kernelWidth; k2++) {
						this.kernels[p][d][k1][k2] = this.kernels[p][d][k1][k2] - learningRate*this.kernelGrad[p][d][k1][k2]/miniBatchSize;
						this.kernelGrad[p][d][k1][k2] = 0;
					}
				}
			}
		}
	}
	
	//sigmoid activation function
	private double actFn(double z) {
		return 1.0/(1.0+Math.pow(Math.E, -z));
	}
}
