/*
 * POOL layer will perform a downsampling operation along the spatial dimensions (width, height), 
 * resulting in smaller spatial dimensions. 
 */
public class PoolingLayer extends Layer{
	private int windowWidth;
	private int windowHeight;
	private int inputDepth;
	private int inputHeight;
	private int inputWidth;
	private int outputDepth;
	private int outputHeight;
	private int outputWidth;
	private int[][][][] winningUnits;
	
	public PoolingLayer(int[] inputDimensions, int[] windowDimensions) throws LayerCompatibilityException {
		//check length of parameters
		if(inputDimensions.length != 3) {
			throw new LayerCompatibilityException("Pooling inputDimensions should be length 3 but was length "+ inputDimensions.length + ".");
		}
		if(windowDimensions.length != 2) {
			throw new LayerCompatibilityException("Pooling windowDimensions should be length 3 but was length "+ windowDimensions.length + ".");
		}
		
		//save dimensions
		this.inputDepth = inputDimensions[0];
		this.inputHeight = inputDimensions[1];
		this.inputWidth = inputDimensions[2];
		this.windowHeight = windowDimensions[0];
		this.windowWidth = windowDimensions[1];
		
		//check if window is compatible with input dimensions
		if(inputWidth%windowWidth != 0 || inputHeight%windowHeight != 0) {
			throw new LayerCompatibilityException("Pooling windowDimensions aren't compatible with its inputDimensions.");
		}
		
		//set output dimensions
		this.outputDepth = this.inputDepth;
		this.outputHeight = this.inputHeight/this.windowHeight;
		this.outputWidth = this.inputWidth/this.windowWidth;
		
		//set inputDim and outputDim for compatibility checking
		this.inputDim[0] = this.inputDepth;
		this.inputDim[1] = this.inputHeight;
		this.inputDim[2] = this.inputWidth;
		this.outputDim[0] = this.outputDepth;
		this.outputDim[1] = this.outputHeight;
		this.outputDim[2] = this.outputWidth;
		
		//create array to hold information on which input activations were the highest
		this.winningUnits = new int[this.outputDepth][this.outputHeight][this.outputWidth][3];
	}
	
	public void feedForward(double[][][] inputActivations) {
		double[][][] outActivations = new double[this.outputDim[0]][this.outputDim[1]][this.outputDim[2]];
		
		//for every window
		for(int d=0; d<this.inputDepth; d++) {	//for each depth slice
			for(int i=0; i<this.inputHeight; i=i+this.windowHeight) {	//every windowHeight rows
				for(int j=0; j<this.inputWidth; j=j+this.windowWidth) {		//every windowWidth columns
					
					//find the max element in the window
					double max = Double.NEGATIVE_INFINITY;
					int[] maxIndices = new int[3];
					for(int m=0; m<this.windowHeight; m++) {
						for(int n=0; n<this.windowWidth; n++) {
							if(inputActivations[d][i+m][j+n] > max) {
								max = inputActivations[d][i+m][j+n];
								maxIndices[0] = d;
								maxIndices[1] = i+m;
								maxIndices[2] = j+n;
							}
						}
					}
					//save maxIndex
					this.winningUnits[d][i/this.windowHeight][j/this.windowWidth][0] = maxIndices[0];
					this.winningUnits[d][i/this.windowHeight][j/this.windowWidth][1] = maxIndices[1];
					this.winningUnits[d][i/this.windowHeight][j/this.windowWidth][2] = maxIndices[2];

					//route max activation
					outActivations[d][i/this.windowHeight][j/this.windowWidth] = max;
				}
			}
		}
		
		if(this.getNextLayer() != null) {
			this.getNextLayer().feedForward(outActivations);
		}
		
	}
	
	public void backpropagate(double[][][] outputErrors) {
		//create array to hold the errors we pass back, initialized to zeros
		double[][][] inDeltas = new double[this.inputDim[0]][this.inputDim[1]][this.inputDim[2]];
		int[] maxIndices = new int[3];
		
		for(int d=0; d<this.outputDepth; d++) {
			for(int i=0; i<this.outputHeight; i++) {
				for(int j=0; j<this.outputWidth; j++) {
					maxIndices[0] = this.winningUnits[d][i][j][0];
					maxIndices[1] = this.winningUnits[d][i][j][1];
					maxIndices[2] = this.winningUnits[d][i][j][2];
					inDeltas[maxIndices[0]][maxIndices[1]][maxIndices[2]] = outputErrors[d][i][j];
				}
			}
		}
		
		//backpropagate the errors
		if(this.getPrevLayer() != null) {
			this.getPrevLayer().backpropagate(inDeltas);
		}
	}
}
