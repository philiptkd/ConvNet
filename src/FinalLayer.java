//assumes the use of a mean square error cost function
public class FinalLayer extends Layer{
	public double[] activations;
	private double[] outDeltas;
	
	public FinalLayer(int length) {
		this.inputLength = length;
		this.outputLength = -1;
		this.activations = new double[length];
		this.outDeltas = new double[length];
	}
	
	//final layer feed forward just saves activations
	public void feedForward(double[] activations) {
		for(int i=0; i<inputLength; i++) {
			this.activations[i] = activations[i];
		}
	}
	
	//calculate delta for the final layer
	public void calcFinalError(int correctClassification) {
		//set the one-hot vector
		int[] y = new int[this.inputLength];	//initializes to all 0s
		y[correctClassification] = 1;
		
		//for mean squared error,
		//	deltaj = (a-y)a(1-a)
		for(int j=0; j<this.inputLength; j++) {
			this.outDeltas[j] = (this.activations[j]-y[j])*this.activations[j]*(1-this.activations[j]);
		}
	}
	
	//final layer backpropagate passes previously calculated deltas to previous layer's backpropagate
	public void backpropagate(double[] outputErrors) {
		if(this.getPrevLayer() != null) {
			this.getPrevLayer().backpropagate(this.outDeltas);
		}
	}
}
