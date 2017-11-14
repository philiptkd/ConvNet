//assumes the use of a mean square error cost function
public class FinalLayer extends Layer{
	public double[] activations;
	private double[][][] deltas;
	
	public FinalLayer(int length) {
		this.inputDim[0] = 1;
		this.inputDim[1] = 1;
		this.inputDim[2] = length;
		this.activations = new double[length];
		this.deltas = new double[inputDim[0]][inputDim[1]][inputDim[2]];
	}
	
	//final layer feed forward just saves activations
	public void feedForward(double[][][] activations) {
		for(int p=0; p<this.inputDim[0]; p++) {
			for(int i=0; i<this.inputDim[1]; i++) {
				for(int j=0; j<this.inputDim[2]; j++) {
					this.activations[p*this.inputDim[1]*this.inputDim[2] + i*this.inputDim[2] + j] = activations[p][i][j];
				}
			}
		}
	}
	
	//calculate delta for the final layer
	public void calcFinalError(int correctClassification) {
		//set the one-hot vector
		int[] y = new int[this.inputDim[2]];	//initializes to all 0s
		y[correctClassification] = 1;
		
		//for mean squared error,
		//	deltaj = (a-y)a(1-a)
		for(int j=0; j<this.inputDim[2]; j++) {
			this.deltas[0][0][j] = (this.activations[j]-y[j])*this.activations[j]*(1-this.activations[j]);
		}
	}
	
	//final layer backpropagate passes previously calculated deltas to previous layer's backpropagate
	public void backpropagate(double[][][] outputErrors) {
		if(this.getPrevLayer() != null) {
			this.getPrevLayer().backpropagate(this.deltas);
		}
	}
}
