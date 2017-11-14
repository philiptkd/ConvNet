//superclass for all the layers
public class Layer {
	public int[] inputDim = new int[3];
	public int[] outputDim = new int[3];
	private Layer prevLayer;
	private Layer nextLayer;

	public void feedForward(double[][][] inputActivations) {
		
	}
	
	public void backpropagate(double[][][] outputErors) {
		
	}
	
	//sets this layer's next layer
	//also calls setPrevLayer on next
	public void setNextLayer(Layer next) throws LayerCompatibilityException {
		if(haveSameDim(next.inputDim, this.outputDim)) {
			this.nextLayer = next;
			next.setPrevLayer(this);
		}
		else if(next.inputDim[0]*next.inputDim[1]*next.inputDim[2] == this.outputDim[0]*this.outputDim[1]*this.outputDim[2])
		{
			System.out.println("Warning: These layers don't have exactly the same dimensions.");
			this.nextLayer = next;
			next.setPrevLayer(this);
		}
		else {
			throw new LayerCompatibilityException("These layers are not compatible.");
		}
	}
	
	//sets this layer's previous layer
	public void setPrevLayer(Layer prev) throws LayerCompatibilityException {
		if(haveSameDim(prev.outputDim, this.inputDim)) {
			this.prevLayer = prev;
		}
		else if(this.inputDim[0]*this.inputDim[1]*this.inputDim[2] == prev.outputDim[0]*prev.outputDim[1]*prev.outputDim[2])
		{
			System.out.println("Warning: These layers don't have exactly the same dimensions.");
			this.prevLayer = prev;
		}
		else {
			throw new LayerCompatibilityException("These layers are not compatible.");
		}
	}
	
	public boolean haveSameDim(int[] dim1, int[] dim2) {
		boolean sameDim = true;
		for(int i=0; i<dim1.length; i++) {
			if(dim1[i] != dim2[i]) {
				sameDim = false;
			}
		}
		return sameDim;
	}
	
	public Layer getNextLayer() {
		return this.nextLayer;
	}
	
	public Layer getPrevLayer() {
		return this.prevLayer;
	}
	
	public void updateWeights(double learningRate, int miniBatchSize) {
		
	}
}
