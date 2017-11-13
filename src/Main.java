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
		InitialLayer L0 = new InitialLayer(28*28);
		
		int[] L1InputDim = {1,28,28};
		int[] L1KernelDim = {6,1,5,5};
		ConvLayer L1 = null;
		
		int[] L2InputDim = {6,24,24};
		int[] L2WindowDim = {2,2};
		PoolingLayer L2 = null;
		
		int[] L3InputDim = {6,12,12};
		int[] L3KernelDim = {12,6,5,5};
		ConvLayer L3 = null;
		
		int[] L4InputDim = {12,8,8};
		int[] L4WindowDim = {2,2};
		PoolingLayer L4 = null;
		
		FullyConnectedLayer L5 = new FullyConnectedLayer(192,10);
		FinalLayer L6 = new FinalLayer(10);
		
		try {
			L1 = new ConvLayer(L1InputDim, L1KernelDim);
			L2 = new PoolingLayer(L2InputDim, L2WindowDim);
			L3 = new ConvLayer(L3InputDim, L3KernelDim);
			L4 = new PoolingLayer(L4InputDim, L4WindowDim);
		} 
		catch (LayerCompatibilityException e) {
			System.out.println(e.getMessage());
		}
				
		//create network
		Layer[] layerList = {L0,L1,L2,L3,L4,L5,L6};
		Network net = new Network(layerList, "mnist_train.csv", "mnist_test.csv", "weights");
		
		try {
			//train net
			net.trainNet(2, 10, 3.0);
			
			//test net
			net.testNet();
		} 
		catch (IOException e) {
			System.out.println(e.getMessage());
		}
		
		
	}

}