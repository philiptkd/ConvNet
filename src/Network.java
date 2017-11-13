import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

public class Network {
	//variables to hold numbers of images
	public int numTrainingImages = 0;
	public int numTestingImages = 0;
	
	//arrays to hold images and labels
	public int[][] trainingImages;
	public int[] trainingLabels;
	private int[][] testingImages;
	private int[] testingLabels;
	
	private Layer[] orderedLayerList;
	private String trainingFileStr;
	private String testingFileStr;
	private String weightsFileStr;
	private InitialLayer inputLayer;
	private FinalLayer outputLayer;
	
	private static Random rand = new Random();
	
	//constructor
	public Network(Layer[] orderedLayerList, String trainingFileStr, String testingFileStr, String weightsFileStr) {
		//save list of layers
		this.orderedLayerList = new Layer[orderedLayerList.length];
		for(int i=0; i<orderedLayerList.length; i++) {
			this.orderedLayerList[i] = orderedLayerList[i];
		}
		//save first and last layers separately for ease of use
		this.inputLayer = (InitialLayer) this.orderedLayerList[0];
		this.outputLayer = (FinalLayer) this.orderedLayerList[orderedLayerList.length-1];
		
		//save file name strings
		this.trainingFileStr = trainingFileStr;
		this.testingFileStr = testingFileStr;
		this.weightsFileStr = weightsFileStr;
		
		//try to connect all the layers in the ordered list of layers
		try {
			for(int i=0; i<orderedLayerList.length-1; i++) {
				orderedLayerList[i].setNextLayer(orderedLayerList[i+1]);
			}
		}
		catch(LayerCompatibilityException e) {
			System.out.println(e.getMessage());
		}
		
		//count the amount of input data and load it in
		this.getNumLines();
		this.loadData();
	}
	
	//do the actual training
	public void trainNet(int epochs, int miniBatchSize, double learningRate) throws IOException {
		//create a list that we can shuffle in order to randomize our mini-batches
		int[] shuffledList = new int[this.numTrainingImages];
		for(int i=0; i<shuffledList.length; i++) {
			shuffledList[i] = i;
		}
		
		for(int epoch=0; epoch<epochs; epoch++) {	//for each epoch
			shuffle(shuffledList);		//shuffle the training set order	
			System.out.println("epoch: " + epoch);
			
			for(int miniBatch=0; miniBatch<shuffledList.length/miniBatchSize; miniBatch++) {	//for each miniBatch
				for(int input=miniBatch*miniBatchSize; input<(miniBatch+1)*miniBatchSize; input++) {	//for each input image
					//load the initial layer
					for(int i=0; i<this.inputLayer.outputLength; i++) {
						this.inputLayer.activations[i] = (double)(this.trainingImages[shuffledList[input]][i])/255.0;	//scale to 0-1
						if(this.inputLayer.activations[i] < 0 || this.inputLayer.activations[i] > 1)		//for troubleshooting. if this is true, something about our input is wrong
						{
							throw new IOException("Input was loaded incorrectly if this is a greyscale image.");
						}
					}
					//this.printInputActivations();
					
					//load correct classification
					int correctClassification = (int)trainingLabels[shuffledList[input]];
					
					//feed forward
					this.inputLayer.feedForward(new double[0]);
					
					//calculate error in last layer
					this.outputLayer.calcFinalError(correctClassification);
					
					//backpropagate
					this.outputLayer.backpropagate(new double[0]);
				}
				//update all the weights and reset the gradients
				for(int i=0; i<this.orderedLayerList.length; i++) {
					this.orderedLayerList[i].updateWeights(learningRate, miniBatchSize);
				}
			}
		}
	}
	
	//get accuracy against the test set
	public void testNet() throws IOException {
		int numCorrect = 0;
		
		for(int image=0; image<this.numTestingImages; image++) {
			//load the input layer
			for(int pixel=0; pixel<this.inputLayer.outputLength; pixel++) {
				this.inputLayer.activations[pixel] = (double)this.testingImages[image][pixel]/255.0;	//scale to 0-1
			}
			
			//feed forward
			this.inputLayer.feedForward(new double[0]);
	
			//load correct classification
			int correctClassification = (int)testingLabels[image];
			
			//see if it classfied correctly
			double highest = Double.NEGATIVE_INFINITY;
			int classification = 0;	//arbitrary
			for(int j=0; j<this.outputLayer.inputLength; j++) {
				if(this.outputLayer.activations[j] > highest) {
					highest = this.outputLayer.activations[j];
					classification = j;
				}
			}
			if(classification == correctClassification) {
				numCorrect = numCorrect + 1;
			}
		}
		
		System.out.println("Accuracy: " + (double)numCorrect/(double)this.numTestingImages);
	}
	
	//gets the numbers of lines in the input files and saves them
	public void getNumLines() {
		try {
			BufferedReader reader = new BufferedReader(new FileReader(this.trainingFileStr));
			while (reader.readLine() != null) this.numTrainingImages++;
			reader.close();
			
			reader = new BufferedReader(new FileReader(this.testingFileStr));
			while (reader.readLine() != null) this.numTestingImages++;
			reader.close();
		}
		catch(Exception e) {
			System.out.println(e.getMessage());
		}
				
		//arrays to hold images and labels
		this.trainingImages = new int[this.numTrainingImages][this.inputLayer.outputLength];
		this.trainingLabels = new int[this.numTrainingImages];
		this.testingImages = new int[this.numTestingImages][this.inputLayer.outputLength];
		this.testingLabels = new int[this.numTestingImages];
	}

	//reads the training and testing data from CSV files and into the static arrays defined above
	//also separates labels and image data
	private void loadData() {
		BufferedReader br = null;
		String line;
		String[] splitLine;
		
		try {
			br = new BufferedReader(new FileReader(this.trainingFileStr));
			for(int i=0; i<this.numTrainingImages; i++) {
				line = br.readLine();
				splitLine = line.split(",");
				this.trainingLabels[i] = Integer.parseInt(splitLine[0]);
				for(int j=0; j<this.trainingImages[0].length; j++) {
					this.trainingImages[i][j] = Integer.parseInt(splitLine[j+1]);
				}
			}
			br.close();
			
			br = new BufferedReader(new FileReader(this.testingFileStr));
			for(int i=0; i<this.numTestingImages; i++) {
				line = br.readLine();
				splitLine = line.split(",");
				this.testingLabels[i] = Integer.parseInt(splitLine[0]);
				for(int j=0; j<this.testingImages[0].length; j++) {
					this.testingImages[i][j] = Integer.parseInt(splitLine[j+1]);
				}
			}
			br.close();
			
		}
		catch(Exception e) {
			System.out.println(e.getMessage());
		}
	}

	//shuffles a list
	private static void shuffle(int[] list) {
		for(int i=list.length-1; i>=0; i--) {
			int index = rand.nextInt(i+1);
			int tmp = list[index];
			list[index] = list[i];
			list[i] = tmp;
		}
	}
	
	public void printInputActivations() {
		for(int i=0; i<28; i++) {
			for(int j=0; j<28; j++) {
				System.out.format("%04f ", this.inputLayer.activations[i*28+j]);
			}
			System.out.println("");
		}
	}
}
