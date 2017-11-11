
public class InitialLayer extends Layer{
	public double[] activations;
	
	public InitialLayer(int length, String dataFile) {
		this.outputLength = length;
		loadData(dataFile);
	}
	
	private void loadData(String dataFile) {
		
	}
	
	public void feedForward(double[] inputActivations) {
		if(this.nextLayer != null) {
			this.nextLayer.feedForward(activations);
		}
	}
}
