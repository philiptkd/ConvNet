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
		FullyConnectedLayer L1 = new FullyConnectedLayer(28*28,30);
		FullyConnectedLayer L2 = new FullyConnectedLayer(30,10);
		FinalLayer L3 = new FinalLayer(10);
		
		//create network
		Layer[] layerList = {L0,L1,L2,L3};
		Network net = new Network(layerList, "mnist_train.csv", "mnist_test.csv", "weights");
		
		try {
			//train
			net.trainNet(30, 10, 3.0);
			
			//test
			net.testNet();
		}
		catch(IOException e) {
			System.out.println(e.getMessage());
		}
		
	}

}