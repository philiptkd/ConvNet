import java.io.File;
import java.io.IOException;

public class Main {
	private static String trainCSVFileString;
	private static String testCSVFileString;
	private static String weightsFileString;
	private static String outputCSVFileString;
	private static int task1;
	private static int task2;
	
	public static void main(String[] args) {
		//check command line arguments
		boolean argsGood = getArgs(args);
		if(!argsGood) {
			return;
		}
		
		Network net = createNet();
		
		try {
			//do task1
			if(task1 == 1) {
				net.trainNet(1, 10, 3.0);
			}
			else {	//task1 = 2
				net.readFromFile();
			}
			
			//do task2
			if(task2 == 3) {
				net.testNet(0); 	//against training data
			}
			else if(task2 == 4) {
				net.testNet(1); 	//against testing data
			}
			else { //task2 = 5
				net.writeToFile();
			}
		}
		catch(IOException e) {
			System.out.println(e.getMessage());
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
		Network net = new Network(layerList, trainCSVFileString, testCSVFileString, weightsFileString, outputCSVFileString);
		return net;
	}

	//to check if the command line arguments are good
	private static boolean getArgs(String[] args) {
		//ensure there are five command line arguments
		if(args.length != 6) {
			System.out.println("This takes 6 arguments: task1, task2, trainCSVFile, testCSVFile, weightsFile, and outputCSVFile");
			System.out.println("[1]: Train Net");
			System.out.println("[2]: Load From File");
			System.out.println("[3]: Print Accuracy on Training Data");
			System.out.println("[4]: Print Accuracy on Test Data");
			System.out.println("[5]: Save to File");
			return false;
		}
		
		//get command line arguments
		trainCSVFileString = args[2];
		testCSVFileString = args[3];
		weightsFileString = args[4];
		outputCSVFileString = args[5];
		
		//see if files exist
		File trainFile = new File(trainCSVFileString);
		File testFile = new File(testCSVFileString);
		File weightsFile = new File(weightsFileString);
		File outputFile = new File(outputCSVFileString);
		
		//build error string to print
		String errorString = "";
		try {
			task1 = Integer.parseInt(args[0]);
			task2 = Integer.parseInt(args[1]);
			
			if(task1 < 1 || task1 > 2) {
				errorString += "task1 should be 1 or 2";
			}
			if(task2 < 3 || task2 > 5) {
				errorString += "task2 should be 3, 4, or 5";
			}
		}
		catch(Error NumberFormatException) {
			errorString += "task1 and task2 should be integers between 1 and 5";
		}
		if(!trainFile.isFile()) {
			errorString += trainCSVFileString + " not found.\n";
		}
		if(!testFile.isFile()) {
			errorString += testCSVFileString + " not found.\n";
		}
		if(!weightsFile.isFile()) {
			try {
				weightsFile.createNewFile();
			} catch (IOException e) {
				System.out.println(e.getMessage());
			}
			//errorString += weightsFileString + " not found.\n";
		}
		if(!outputFile.isFile()) {
			try {
				outputFile.createNewFile();
			} catch (IOException e) {
				System.out.println(e.getMessage());
			}
			//errorString += outputCSVFileString + " not found.\n";
		}
		
		if(errorString != "") {
			System.out.println(errorString);
			return false;
		}
		
		return true;
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
		Network net = new Network(layerList, "mnist_train.csv", "mnist_test.csv", "weights", "output.csv");
		
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