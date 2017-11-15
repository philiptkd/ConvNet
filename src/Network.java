import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.RandomAccessFile;
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
	private String outputFileStr;
	private InitialLayer inputLayer;
	private FinalLayer outputLayer;
	
	private static Random rand = new Random();
	
	//constructor
	public Network(Layer[] orderedLayerList, String trainingFileStr, String testingFileStr, String weightsFileStr, String outputFileStr) {
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
		this.outputFileStr = outputFileStr;
		
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
			System.out.print("epoch " + epoch + ": ");
			
			for(int miniBatch=0; miniBatch<shuffledList.length/miniBatchSize; miniBatch++) {	//for each miniBatch
				for(int input=miniBatch*miniBatchSize; input<(miniBatch+1)*miniBatchSize; input++) {	//for each input image
					//load the initial layer
					for(int i=0; i<this.inputLayer.outputDim[2]; i++) {
						this.inputLayer.activations[0][0][i] = (double)(this.trainingImages[shuffledList[input]][i])/255.0;	//scale to 0-1
						if(this.inputLayer.activations[0][0][i] < 0 || this.inputLayer.activations[0][0][i] > 1)		//for troubleshooting. if this is true, something about our input is wrong
						{
							throw new IOException("Input was loaded incorrectly if this is a greyscale image.");
						}
					}
					
					//load correct classification
					int correctClassification = (int)trainingLabels[shuffledList[input]];
					
					//feed forward
					this.inputLayer.feedForward(new double[0][0][0]);
					
					//calculate error in last layer
					this.outputLayer.calcFinalError(correctClassification);
					
					//backpropagate
					this.outputLayer.backpropagate(new double[0][0][0]);
				}
				//update all the weights and reset the gradients
				for(int i=0; i<this.orderedLayerList.length; i++) {
					this.orderedLayerList[i].updateWeights(learningRate, miniBatchSize);
				}
			}
			//print accuracy against training data after each epoch
			//this.testNet(0);
		}
	}
	
	//get accuracy against either the training set or the test set
	public void testNet(int dataSet) throws IOException {
		int numImages;
		int[][] images;
		int[] labels;
		
		//test against training data
		if(dataSet == 0) {
			numImages = numTrainingImages;
			images = trainingImages;
			labels = trainingLabels;
		}
		//test against testing data
		else {	//dataSet = 1
			numImages = numTestingImages;
			images = testingImages;
			labels = testingLabels;
		}
		
		int numCorrect = 0;
		
		for(int image=0; image<numImages; image++) {
			//load the input layer
			for(int pixel=0; pixel<this.inputLayer.outputDim[2]; pixel++) {
				this.inputLayer.activations[0][0][pixel] = (double)images[image][pixel]/255.0;	//scale to 0-1
			}
			
			//feed forward
			this.inputLayer.feedForward(new double[0][0][0]);
	
			//load correct classification
			int correctClassification = (int)labels[image];
			
			//see if it classfied correctly
			double highest = Double.NEGATIVE_INFINITY;
			int classification = 0;	//arbitrary
			for(int j=0; j<this.outputLayer.inputDim[2]; j++) {
				if(this.outputLayer.activations[j] > highest) {
					highest = this.outputLayer.activations[j];
					classification = j;
				}
			}
			if(classification == correctClassification) {
				numCorrect = numCorrect + 1;
			}
		}
		
		//System.out.println(100*(double)numCorrect/(double)numImages + "% accurate.");
		FileWriter writer = new FileWriter(outputFileStr, true);
		writer.write(Double.toString((double)numCorrect/(double)numImages) + "\n");
		writer.close();
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
		this.trainingImages = new int[this.numTrainingImages][this.inputLayer.outputDim[2]];
		this.trainingLabels = new int[this.numTrainingImages];
		this.testingImages = new int[this.numTestingImages][this.inputLayer.outputDim[2]];
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
	
	//used to save network parameters to a file
	public void writeToFile() {
		RandomAccessFile writer = null;
		try {
			writer = new RandomAccessFile(this.weightsFileStr, "rw");

			//write number of FC or Conv layers
			int numImportantLayers = 0;
			for(int i=0; i<this.orderedLayerList.length; i++) {
				if(this.orderedLayerList[i] instanceof FullyConnectedLayer ||
						this.orderedLayerList[i] instanceof ConvLayer) {
					numImportantLayers++;
				}
			}
			writer.writeDouble((double)numImportantLayers);
			
			//for each layer
			for(int i=0; i<this.orderedLayerList.length; i++) {
				Layer theLayer = this.orderedLayerList[i];
				
				//if it's a FC layer
				if(theLayer instanceof FullyConnectedLayer) {
					//write type of layer
					writer.writeDouble(0.0); 	
					
					//write dimensions of weights
					int outputLength = theLayer.outputDim[2];
					int inputLength = theLayer.inputDim[2];
					writer.writeDouble((double)outputLength);	
					writer.writeDouble((double)inputLength);
					
					//write weights
					for(int j=0; j<outputLength; j++) {
						for(int k=0; k<inputLength; k++) {
							writer.writeDouble(((FullyConnectedLayer) theLayer).weights[j][k]);
						}
					}
					
					//write dimensions of biases
					writer.writeDouble((double)outputLength);
					
					//write biases
					for(int j=0; j<outputLength; j++) {
						writer.writeDouble(((FullyConnectedLayer) theLayer).outBiases[j]);
					}
				}
				
				//if it's a convolutional layer
				else if(theLayer instanceof ConvLayer) {
					//write type of layer
					writer.writeDouble(1.0); 	
					
					//write dimensions of weights
					int numFilters = ((ConvLayer) theLayer).numFilters;
					int kernelDepth = ((ConvLayer) theLayer).kernelDepth;
					int kernelHeight = ((ConvLayer) theLayer).kernelHeight;
					int kernelWidth = ((ConvLayer) theLayer).kernelWidth;
					writer.writeDouble((double)numFilters);
					writer.writeDouble((double)kernelDepth);
					writer.writeDouble((double)kernelHeight);
					writer.writeDouble((double)kernelWidth);
					
					//write weights
					for(int q=0; q<numFilters; q++) {
						for(int p=0; p<kernelDepth; p++) {
							for(int m=0; m<kernelHeight; m++) {
								for(int n=0; n<kernelWidth; n++) {
									writer.writeDouble(((ConvLayer) theLayer).kernels[q][p][m][n]);
								}
							}
						}
					}
					
					//write dimensions of biases
					writer.writeDouble(numFilters);
					
					//write biases
					for(int q=0; q<numFilters; q++) {
						writer.writeDouble(((ConvLayer) theLayer).outBiases[q]);
					}
				}
			}
			
			writer.close();
		}
		catch(Exception e) {
			System.out.println(e.getMessage());
		}
	}
	
	//used to load saved network parameters from a file
	public boolean readFromFile() {
		boolean successfulRead = false;
		RandomAccessFile reader = null;
		try {
			reader = new RandomAccessFile(this.weightsFileStr, "r");
			
			//read number of important layers in file
			int numLayers = (int)reader.readDouble();
			
			//check against number of layers in network
			int numLayersInNet = 0;
			for(int i=0; i<this.orderedLayerList.length; i++) {
				if(this.orderedLayerList[i] instanceof FullyConnectedLayer ||
						this.orderedLayerList[i] instanceof ConvLayer) {
					numLayersInNet++;
				}
			}
			
			//return if they're not the same
			if(numLayers != numLayersInNet) {
				stopReading(reader);
				return successfulRead;
			}
			
			//get the first layer of the network
			Layer theLayer = this.orderedLayerList[0];
			
			//for each important layer
			for(int i=0; i<numLayers; i++) {
				//get type of layer
				int layerType = (int)reader.readDouble();
				
				//get next important layer in network
				while(theLayer != null &&
						!(theLayer instanceof FullyConnectedLayer) && 
						!(theLayer instanceof ConvLayer)) {
					theLayer = theLayer.getNextLayer();
				}
				
				//if it's a FC layer
				if(layerType == 0 && theLayer instanceof FullyConnectedLayer) {
					//read the two weights dimensions
					int outputLength = (int)reader.readDouble();
					int inputLength = (int)reader.readDouble();
					
					//check against weights dimensions of network layer
					if(theLayer.outputDim[2] != outputLength || theLayer.inputDim[2] != inputLength) {
						stopReading(reader);
						return successfulRead;
					}
					
					//read the weights
					for(int j=0; j<outputLength; j++) {
						for(int k=0; k<inputLength; k++) {
							((FullyConnectedLayer) theLayer).weights[j][k] = reader.readDouble();
						}
					}
					
					//read the one biases dimension
					if((int)reader.readDouble() != outputLength) {
						stopReading(reader);
						return successfulRead;
					}
					
					//read the biases
					for(int j=0; j<outputLength; j++) {
						((FullyConnectedLayer) theLayer).outBiases[j] = reader.readDouble();
					}
				}
				
				//if it's a convolutional layer
				else if(layerType == 1 && theLayer instanceof ConvLayer) {
					//read the four weights dimensions
					int numFilters = (int)reader.readDouble();
					int kernelDepth = (int)reader.readDouble();
					int kernelHeight = (int)reader.readDouble();
					int kernelWidth = (int)reader.readDouble();
					
					//check against weights dimensions of network layer
					if(((ConvLayer) theLayer).numFilters != numFilters ||
							((ConvLayer) theLayer).kernelDepth != kernelDepth ||
							((ConvLayer) theLayer).kernelHeight != kernelHeight ||
							((ConvLayer) theLayer).kernelWidth != kernelWidth){
						stopReading(reader);
						return successfulRead;
					}
					
					//read the weights
					for(int q=0; q<numFilters; q++) {
						for(int p=0; p<kernelDepth; p++) {
							for(int m=0; m<kernelHeight; m++) {
								for(int n=0; n<kernelWidth; n++) {
									((ConvLayer) theLayer).kernels[q][p][m][n] = reader.readDouble();
								}
							}
						}
					}
					
					//read the one biases dimension
					if(reader.readDouble() != numFilters) {
						stopReading(reader);
						return successfulRead;
					}
					
					//read the biases
					for(int q=0; q<numFilters; q++) {
						((ConvLayer) theLayer).outBiases[q] = reader.readDouble();
					}
				}
				
				else {
					stopReading(reader);
					return successfulRead;
				}
				
				//go to next layer
				theLayer = theLayer.getNextLayer();
			}
			
			reader.close();
			successfulRead = true;
		}
		catch(Exception e) {
			System.out.println(e.getMessage());
		}
		
		return successfulRead;
	}
	
	private void stopReading(RandomAccessFile reader) {
		System.out.println("This weights file is incompatible with the current network.");
		try {
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
