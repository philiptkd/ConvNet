
public class InitialLayer extends Layer{
	public double[][][] activations;
	
	public InitialLayer(int length) {
		this.outputDim[0] = 1;
		this.outputDim[1] = 1;
		this.outputDim[2] = length;
		this.activations = new double[this.outputDim[0]][this.outputDim[1]][this.outputDim[2]];
	}
	
	public void feedForward(double[][][] inputActivations) {
		if(this.getNextLayer() != null) {
			this.getNextLayer().feedForward(activations);
		}
	}
}
