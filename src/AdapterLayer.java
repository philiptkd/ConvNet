//to adapt the shape of the matrices going forward and backward through the network
public class AdapterLayer extends Layer{
	public AdapterLayer(int[] inputDim, int[] outputDim) {
		//save dimensions
		for(int i=0; i<3; i++) {
			this.inputDim[i] = inputDim[i];
			this.outputDim[i] = outputDim[i];
		}
	}
	
	public void feedForward(double[][][] inputActivations) {
		//continue feed forward
		if(this.getNextLayer() != null) {
			this.getNextLayer().feedForward(adapter(inputActivations, this.inputDim, this.outputDim));
		}
	}
	
	public void backpropagate(double[][][] outputErrors) {
		//continue backpropagate
		if(this.getPrevLayer() != null) {
			this.getPrevLayer().backpropagate(adapter(outputErrors, this.outputDim, this.inputDim));
		}
	}
	
	private double[][][] adapter(double[][][] inArray, int[] oldDims, int[] newDims) {
		double[][][] outArray = new double[newDims[0]][newDims[1]][newDims[2]];
		int length = newDims[0]*newDims[1]*newDims[2];
		double[] tmp = new double[length];
		
		//put in 1d array
		for(int p=0; p<oldDims[0]; p++) {
			for(int i=0; i<oldDims[1]; i++) {
				for(int j=0; j<oldDims[2]; j++) {
					tmp[p*oldDims[1]*oldDims[2] + i*oldDims[2] + j] = inArray[p][i][j];
				}
			}
		}
		
		//put in compatible array
		for(int q=0; q<newDims[0]; q++) {
			for(int m=0; m<newDims[1]; m++) {
				for(int n=0; n<newDims[2]; n++) {
					outArray[q][m][n] = tmp[q*newDims[1]*newDims[2] + m*newDims[2] + n];
				}
			}
		}
		
		return outArray;
	}
}
