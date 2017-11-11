/*
 * TODO: 
 * modular architecture
 * layers as objects
 * 		input and output 3d matrices
 * 		feed forward and backpropagate functions
 * 
 */
public class Main {

	public static void main(String[] args) {
		//create layers
		InitialLayer L0 = new InitialLayer(1,"str");
		FullyConnectedLayer L1 = new FullyConnectedLayer(1,2);
		FullyConnectedLayer L2 = new FullyConnectedLayer(2,3);
		FullyConnectedLayer L3 = new FullyConnectedLayer(3,4);
		FinalLayer L4 = new FinalLayer(4);
	
		//try to connect them
		try {
			L0.setNextLayer(L1);
			L1.setNextLayer(L2);
			L2.setNextLayer(L3);
			L3.setNextLayer(L4);
		}
		catch(LayerCompatibilityException e) {
			System.out.println(e.getMessage());
		}
		
		//initialize input layer for testing
		L0.activations = new double[1];
		L0.activations[0] = 1;
		
		//feed forward through the whole network, using dummy parameter
		L0.feedForward(new double[1]);
		
		//print network output
		for(int i=0; i<L4.activations.length; i++) {
			System.out.println(L4.activations[i]);
		}
	}

}