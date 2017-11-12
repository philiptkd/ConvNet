
public class InitialLayer extends Layer{
	public double[] activations;
	
	public InitialLayer(int length) {
		this.outputLength = length;
		this.inputLength = -1;
		this.activations = new double[length];
	}
	
	public void feedForward(double[] inputActivations) {
		if(this.getNextLayer() != null) {
			this.getNextLayer().feedForward(activations);
		}
	}
}
