//superclass for all the layers
public class Layer {
	public int inputLength;
	public int outputLength;
	public Layer prevLayer;
	public Layer nextLayer;

	public void feedForward(double[] inputActivations) {
		
	}
	
	//sets this layer's next layer
	//also calls setPrevLayer on next
	public void setNextLayer(Layer next) throws LayerCompatibilityException {
		if(next.inputLength == this.outputLength) {
			this.nextLayer = next;
			next.setPrevLayer(this);
		}
		else {
			throw new LayerCompatibilityException("The requested next layer has " + next.inputLength + " input nodes, but this layer has " + this.outputLength + " output nodes.");
		}
	}
	
	//sets this layer's previous layer
	public void setPrevLayer(Layer prev) throws LayerCompatibilityException {
		if(prev.outputLength == this.inputLength) {
			this.prevLayer = prev;
		}
		else {
			throw new LayerCompatibilityException("The requested previous layer has " + prev.outputLength + " output nodes, but this layer has " + this.inputLength + " input nodes.");
		}
	}
}
