/*
 * RELU layer will apply an elementwise activation function, such as the max(0,x) thresholding 
 * at zero. This leaves the size of the volume unchanged.
 */
public class ReLuLayer extends Layer{
	private double[] inputActivations;
	
	//constructor
	public ReLuLayer(int length) {
		this.inputLength = length;
		this.outputLength = length;
		
		this.inputActivations = new double[this.inputLength];
	}
	
	public void feedForward(double[] inActivations) {
		//save input activations
		for(int i=0; i<this.inputLength; i++) {
			this.inputActivations[i] = inActivations[i];
		}
		
		//create array to pass activations with
		double[] outActivations = new double[this.outputLength];
		
		//pass activations if they are greater than zero
		for(int i=0; i<this.inputLength; i++) {
			if(this.inputActivations[i] > 0) {
				outActivations[i] = this.inputActivations[i];
			}
		}
		
		//continue feeding forward
		if(this.getNextLayer() != null) {
			this.getNextLayer().feedForward(outActivations);
		}
	}
	
	public void backpropagate(double[] outputErrors) {
		//create array to pass errors with, initialized to zeros
		double[] inDeltas = new double[this.inputLength];
		
		//pass error if the activation was originally greater than zero
		for(int i=0; i<this.outputLength; i++) {
			if(this.inputActivations[i] > 0) {
				inDeltas[i] = outputErrors[i];
			}
		}
		
		//backpropagate the error
		if(this.getPrevLayer() != null) {
			this.getPrevLayer().backpropagate(inDeltas);
		}
	}
}
