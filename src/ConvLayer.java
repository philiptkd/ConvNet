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
		
		//save in inputDim and outputDim for compatibility checking
		this.inputDim[0] = this.inputDepth;
		this.inputDim[1] = this.inputHeight;
		this.inputDim[2] = this.inputWidth;
		this.outputDim[0] = this.outputDepth;
		this.outputDim[1] = this.outputHeight;
		this.outputDim[2] = this.outputWidth;
		
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
						this.kernels[p][d][k1][k2] = rand.nextGaussian()/Math.sqrt(this.kernelHeight*this.kernelWidth);
					}
				}
			}
		}
		
		//the initial variance of the biases matters less
		for(int i=0; i<this.outBiases.length; i++) {
			this.outBiases[i] = rand.nextGaussian();
		}
	}
	
	public void feedForward(double[][][] inputActivations) {
		//save input activations in multidimensional array
		for(int d=0; d<this.inputDepth; d++) {
			for(int h=0; h<this.inputHeight; h++) {
				for(int w=0; w<this.inputWidth; w++) {
					this.inActivations[d][h][w] = inputActivations[d][h][w];
				}
			}
		}
		
		//create array to hold and pass output activations 
		double[][][] outActivations = new double[this.outputDepth][this.outputHeight][this.outputWidth];
		
		//for each output pixel
		for(int f=0; f<this.outputDepth; f++) {	//for each filter
			for(int i=0; i<this.outputHeight; i++) { //for each output row
				for(int j=0; j<this.outputWidth; j++) { //for each output column
					
					//calculate the dot product of the kernel with the input
					double tmpZ = 0;
					for(int d=0; d<this.kernelDepth; d++) {	//for each kernel depth slice
						for(int m=0; m<this.kernelHeight; m++) { //for each kernel row
							for(int n=0; n<this.kernelWidth; n++) { //for each kernel column
								tmpZ += this.inActivations[d][i+m][j+n]*this.kernels[f][d][m][n];
							}
						}
					}
					
					//add the bias
					tmpZ += this.outBiases[f];
					
					//save
					this.outZs[f][i][j] = tmpZ;
					
					//calculate output activation
					outActivations[f][i][j] = actFn(tmpZ);
				}
			}
		}
		
		//pass activations to next layer
		if(this.getNextLayer() != null) {
			this.getNextLayer().feedForward(outActivations);
		}
	}
	
	public void backpropagate(double[][][] outputErrors) {
		//save output delta in multidimensional array
		for(int d=0; d<this.outputDepth; d++) {
			for(int h=0; h<this.outputHeight; h++) {
				for(int w=0; w<this.outputWidth; w++) {
					this.outDeltas[d][h][w] = outputErrors[d][h][w];
				}
			}
		}
		
		//create array to hold and pass the input errors
		double[][][] inDeltas = new double[this.inputDepth][this.inputHeight][this.inputWidth];
		
		//for each input pixel
		for(int p=0; p<this.inputDepth; p++) {	//for each input/kernel depth slice
			for(int m=0; m<this.inputHeight; m++) {	//for each input row
				for(int n=0; n<this.inputWidth; n++) {	//for each input column
					
					//calculate the dot product of the flipped kernel with the output errors
					double tmpD = 0;
					for(int q=0; q<this.outputDepth; q++) {
						for(int i=0; i<this.outputHeight; i++) {
							for(int j=0; j<this.outputWidth; j++) {
								tmpD += this.outDeltas[q][i][j]*this.kernels[q][p][m-i][n-j];
							}
						}
					}
					
					//multiply by sigma prime
					tmpD = tmpD*this.inActivations[p][m][n]*(1 - this.inActivations[p][m][n]);
					
					//save
					inDeltas[p][m][n] = tmpD;
				}
			}
		}
		
		//increment bias gradients for the output layer
		//each bias contributes to a whole depth slice of output errors
		for(int q=0; q<this.numFilters; q++) {
			for(int i=0; i<this.outputHeight; i++) {
				for(int j=0; j<this.outputWidth; j++) {
					this.outBiasesGrad[q] += this.outDeltas[q][i][j];
				}
			}
		}
		
		//increment weight gradients
		for(int q=0; q<this.numFilters; q++) {
			for(int p=0; p<this.kernelDepth; p++) {
				for(int u=0; u<this.kernelHeight; u++) {
					for(int v=0; v<this.kernelWidth; v++) {
						
						//sum over the output pixels that this weight affected
						for(int i=0; i<this.outputHeight; i++) {
							for(int j=0; j<this.outputWidth; j++) {
								this.kernelGrad[q][p][u][v] += this.outDeltas[q][i][j]*this.inActivations[p][i+u][j+v];
							}
						}
					}
				}
			}
		}
		
		//continue backpropagating
		if(this.getPrevLayer() != null) {
			this.getPrevLayer().backpropagate(inDeltas);
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
