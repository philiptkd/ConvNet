import java.io.IOException;

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
		InitialLayer L0 = new InitialLayer(3*3);
		
		int[] L1InputDim = {1,3,3};
		int[] L1KernelDim = {1,1,2,2};
		ConvLayer L1 = null;
		
		try {
			L1 = new ConvLayer(L1InputDim, L1KernelDim);
		} 
		catch (LayerCompatibilityException e) {
			System.out.println(e.getMessage());
		}
		
		FinalLayer L2 = new FinalLayer(4);
		
		//create network
		Layer[] layerList = {L0,L1,L2};
		Network net = new Network(layerList, "mnist_train.csv", "mnist_test.csv", "weights");
		
		//build toy network
		int val = 1;
		for(int i=0; i<L0.activations.length; i++) {
			L0.activations[i] = (double)val;
			val = val^1;
		}
		if(L1 != null) {
			L1.kernels[0][0][0][0] = 1;
			L1.kernels[0][0][0][1] = 0;
			L1.kernels[0][0][1][0] = 1;
			L1.kernels[0][0][1][1] = 0;
			L1.outBiases[0] = 0;
		}
		
		//feed forward
		L0.feedForward(new double[0]);
	}

}