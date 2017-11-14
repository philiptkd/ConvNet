/*
 * RELU layer will apply an elementwise activation function, such as the max(0,x) thresholding 
 * at zero. This leaves the size of the volume unchanged.
 */
public class ReLuLayer extends Layer{
	private double[][][] inputActivations;
	
	//constructor
	public ReLuLayer(int[] dims) {
		for(int i=0; i<3; i++) {
			this.inputDim[i] = dims[i];
			this.outputDim[i] = dims[i];
		}
		
		this.inputActivations = new double[this.inputDim[0]][this.inputDim[1]][this.inputDim[2]];
	}
	
	public void feedForward(double[][][] inActivations) {
		//save input activations
		for(int p=0; p<this.inputDim[0]; p++) {
			for(int i=0; i<this.inputDim[1]; i++) {
				for(int j=0; j<this.inputDim[2]; j++) {
					this.inputActivations[p][i][j] = inActivations[p][i][j];
				}
			}
		}
		
		//create array to pass activations with
		double[][][] outActivations = new double[this.outputDim[0]][this.outputDim[1]][this.outputDim[2]];
		
		//pass activations if they are greater than zero
		for(int p=0; p<this.inputDim[0]; p++) {
			for(int i=0; i<this.inputDim[1]; i++) {
				for(int j=0; j<this.inputDim[2]; j++) {
					outActivations[p][i][j] = this.inputActivations[p][i][j];
				}
			}
		}
		
		//continue feeding forward
		if(this.getNextLayer() != null) {
			this.getNextLayer().feedForward(outActivations);
		}
	}
	
	public void backpropagate(double[][][] outputErrors) {
		//create array to pass errors with, initialized to zeros
		double[][][] inDeltas = new double[this.inputDim[0]][this.inputDim[1]][this.inputDim[2]];
		
		//pass error if the activation was originally greater than zero
		for(int q=0; q<this.outputDim[0]; q++) {
			for(int m=0; m<this.outputDim[1]; m++) {
				for(int n=0; n<this.outputDim[2]; n++) {
					if(this.inputActivations[q][m][n] > 0) {
						inDeltas[q][m][n] = outputErrors[q][m][n];
					}
				}
			}
		}
		
		//backpropagate the error
		if(this.getPrevLayer() != null) {
			this.getPrevLayer().backpropagate(inDeltas);
		}
	}
}
