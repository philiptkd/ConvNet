
public class FinalLayer extends Layer{
	public double[] activations;
	
	public FinalLayer(int length) {
		this.inputLength = length;
		this.activations = new double[length];
	}
	
	public void feedForward(double[] activations) {
		for(int i=0; i<inputLength; i++) {
			this.activations[i] = activations[i];
		}
	}
}
