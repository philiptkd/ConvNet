import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class Main {

	public static void main(String[] args) {
		boolean trained = false;
		
		//print the intro
		System.out.println("This is a program that demonstrates a convolutional neural network that classifies handwritten digits from the MNIST data set. " + 
				"\nIt uses stochastic gradient descent with backpropagation to train the network."
				+ "\nThe current network architecture is this: "
				+ "\nConvLayer -> PoolingLayer -> ConvLayer -> PoolingLayer -> FCLayer."
				+ "\n\nThe network may take a second or two to be set up.\n\n");
		
		//create the network
		Network net = createNet();
		
		while(true) {
			//print the user input options
			System.out.println("Choose from the options below.\n");
			System.out.println("[1] Train network");
			System.out.println("[2] Load pre-trained network");
			if(trained) {
				System.out.println("[3] Display network accuracy on training data");
				System.out.println("[4] Display network accuracy on testing data");
				System.out.println("[5] Save network state to file");
			}
			System.out.println("[6] Exit");
			
			//wait for user input
			int selection = 0;
			BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
			try {
				selection = Integer.parseInt(br.readLine());
			} catch (Exception e) {
				System.out.println("Something went wrong. Please choose from the options below.");
			}
			
			//switch on user input
			try {
				switch(selection) {
				case 0:	//do nothing
					break;
				case 1:	//train network
					net.trainNet(3, 10, 3.0);
					trained = true;
					break;
				case 2:	//load pre-trained network
					boolean successfulRead = net.readFromFile();
					if(successfulRead) {
						trained = true;
					}
					break;
				case 3: //display network accuracy on training data
					if(trained) {
						net.testNet(0);
					}
					break;
				case 4: //display network accuracy on testing data
					if(trained) {
						net.testNet(1);
					}
					break;
				case 5: //save network state to file
					if(trained) {
						net.writeToFile();
					}
					break;
				case 6: //exit
					return;
				}
			}
			catch (IOException e) {
				System.out.println(e.getMessage());
			}
		}
	}
	
	public static Network createNet() {
		InitialLayer L0 = new InitialLayer(28*28);
		
		int[] A1InDim = {1,1,28*28};
		int[] A1OutDim = {1,28,28};
		AdapterLayer A1 = new AdapterLayer(A1InDim, A1OutDim);
		
		int[] L2InDim = {1,28,28};
		int[] L2KernelDim = {6,1,5,5};
		ConvLayer L2 = null;
		
		int[] L3InDim = {6,24,24};
		int[] L3WinDim = {2,2};
		PoolingLayer L3 = null;
		
		int[] L4InDim = {6,12,12};
		int[] L4KernelDim = {12,6,5,5};
		ConvLayer L4 = null;
		
		int[] L5InDim = {12,8,8};
		int[] L5WinDim = {2,2};
		PoolingLayer L5 = null;
		
		int[] A6InDim = {12,4,4};
		int[] A6OutDim = {1,1,192};
		AdapterLayer A6 = new AdapterLayer(A6InDim, A6OutDim);
		
		FullyConnectedLayer L7 = new FullyConnectedLayer(192, 10);
		
		FinalLayer L8 = new FinalLayer(10);
		
		try {
			L2 = new ConvLayer(L2InDim, L2KernelDim);
			L3 = new PoolingLayer(L3InDim, L3WinDim);
			L4 = new ConvLayer(L4InDim, L4KernelDim);
			L5 = new PoolingLayer(L5InDim, L5WinDim);
		}
		catch(LayerCompatibilityException e) {
			System.out.println(e.getMessage());
		}
		
		Layer[] layerList = {L0,A1,L2,L3,L4,L5,A6,L7,L8};
		Network net = new Network(layerList, "mnist_train.csv", "mnist_test.csv", "weights");
		return net;
	}
	
	public static void toyNet() {
		//create layers
		InitialLayer L0 = new InitialLayer(3*5*5);
		
		int[] A1InputDim = {1,1,5*5*3};
		int[] A1OutputDim = {3,5,5};
		AdapterLayer A1 = new AdapterLayer(A1InputDim,A1OutputDim);
		
		int[] L2InputDim = {3,5,5};
		int[] L2KernelDim = {2,3,3,3};
		ConvLayer L2 = null;
		
		int[] L3InputDim = {2,3,3};
		int[] L3KernelDim = {3,2,2,2};
		ConvLayer L3 = null;
		
		int[] A4InputDim = {3,2,2};
		int[] A4OutputDim = {1,1,3*2*2};
		AdapterLayer A4 = new AdapterLayer(A4InputDim, A4OutputDim); 
		
		FinalLayer L5 = new FinalLayer(3*2*2);
		
		try {
			L2 = new ConvLayer(L2InputDim, L2KernelDim);
			L3 = new ConvLayer(L3InputDim, L3KernelDim);
		}
		catch(LayerCompatibilityException e) {
			System.out.println(e.getMessage());
		}
		
		//create network
		Layer[] layerList = {L0,A1,L2,L3,A4,L5};
		Network net = new Network(layerList, "mnist_train.csv", "mnist_test.csv", "weights");
		
		//create toy input
		double[][][] toyInput = {{{1,0,1,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1,1,0,0,1,1,0,1,0,1,0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,1,1,0,0,0,1,0,0,1,1,0,1,0,0,0,0,1,1,0,0,0,1}}};
		L0.activations = toyInput;
		L0.feedForward(new double[0][0][0]);
		
		//create toy kernels
		double[][][][] K1 = {{{{0,1,1},{-1,-1,-1},{0,-1,0}},{{0,-1,1},{1,0,1},{1,0,1}},{{0,0,1},{-1,1,0},{0,1,0}}},  {{{1,0,0},{-1,1,1},{0,-1,-1}},{{1,1,-1},{-1,-1,1},{1,-1,0}},{{-1,1,0},{0,0,1},{-1,-1,-1}}}};
		L2.kernels = K1;
		double[][][][] K2 = {{{{1,-1},{-1,0}},{{1,-1},{0,1}}},   {{{-1,0},{-1,-1}},{{0,-1},{-1,0}}},   {{{0,1},{1,-1}},{{1,0},{0,0}}}};
		L3.kernels = K2;
		
		L0.feedForward(new double[0][0][0]);
	}

}