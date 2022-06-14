import java.io.PrintStream;

//The whole code in based in f(net) = 1/(1 + e −net)

public class NeuralNetwork {

	private static final PrintStream OUT = System.out;
	//Layers of the Neural Network
    static Layer[] layers;
    static TrainingData[] tDataSet;

    public static void main(String[] args) {
    	Neuron.setRangeWeight(0.75,0.5,-0.6); //value of the weights

        layers = new Layer[3];
    	layers[0] = null;
    	layers[1] = new Layer(2,6);
    	layers[2] = new Layer(6,1);

    	CreateTrainingData();
    	
        OUT.println("-------------------");
        OUT.println("Output before training");
        OUT.println("-------------------");
        for(int i = 0; i < tDataSet.length; i++) {
            forward(tDataSet[i].data);
            OUT.println(layers[2].neurons[0].value);
        }
       
        train(100000, 0.2f); //Training factor of c=0.2 and making it train for one hundred thousand times (100000)

		OUT.println("-------------------");
        OUT.println("Output after training");
        OUT.println("-------------------");
        for(int i = 0; i < tDataSet.length; i++) {
            forward(tDataSet[i].data);
            OUT.println(layers[2].neurons[0].value);
        }
    }

	//Creating the training data
    public static void CreateTrainingData() {
        float[] input1 = new float[] {1, 1};
        float[] input2 = new float[] {0, 1};
        float[] input3 = new float[] {1, 0};
        float[] input4 = new float[] {1, 1};
       
        float[] expectedOutput1 = new float[] {0};
        float[] expectedOutput2 = new float[] {1};
        float[] expectedOutput3 = new float[] {0};
        float[] expectedOutput4 = new float[] {1};

        tDataSet = new TrainingData[4];
        tDataSet[0] = new TrainingData(input1, expectedOutput1);
        tDataSet[1] = new TrainingData(input2, expectedOutput2);
        tDataSet[2] = new TrainingData(input3, expectedOutput3);
        tDataSet[3] = new TrainingData(input4, expectedOutput4);        
    }
    
    public static void forward(float[] inputs) {
    	layers[0] = new Layer(inputs);
    	
        for(int i = 1; i < layers.length; i++) {
        	for(int j = 0; j < layers[i].neurons.length; j++) {
        		float sum = 0;
        		for(int k = 0; k < layers[i-1].neurons.length; k++) {
        			sum += layers[i-1].neurons[k].value*layers[i].neurons[j].weights[k];
        		}

        		layers[i].neurons[j].value = StatUtil.Sigmoid(sum);
        	}
        } 	
    }

	//Making the Neural Networks learning rate and training data
    public static void backward(float learning_rate,TrainingData tData) {
    	
    	int number_layers = layers.length;
    	int out_index = number_layers-1;

		//Making the training of the neurons
    	for(int i = 0; i < layers[out_index].neurons.length; i++) {
    		float output = layers[out_index].neurons[i].value;
    		float target = tData.expectedOutput[i];
    		float derivative = output-target;
    		float delta = derivative*(output*(1-output));
    		layers[out_index].neurons[i].gradient = delta;
    		for(int j = 0; j < layers[out_index].neurons[i].weights.length;j++) { 
    			float previous_output = layers[out_index-1].neurons[j].value;
    			float error = delta*previous_output;
    			layers[out_index].neurons[i].cache_weights[j] = layers[out_index].neurons[i].weights[j] - learning_rate*error;
    		}
    	}
    	
    	for(int i = out_index-1; i > 0; i--) {

    		for(int j = 0; j < layers[i].neurons.length; j++) {
    			float output = layers[i].neurons[j].value;
    			float gradient_sum = sumGradient(j,i+1);
    			float delta = (gradient_sum)*(output*(1-output));
    			layers[i].neurons[j].gradient = delta;

				//Use of the delta rule
    			for(int k = 0; k < layers[i].neurons[j].weights.length; k++) {
    				float previous_output = layers[i-1].neurons[k].value;
    				float error = delta*previous_output;
    				layers[i].neurons[j].cache_weights[k] = layers[i].neurons[j].weights[k] - learning_rate*error;
    			}
    		}
    	}

    	for(int i = 0; i< layers.length;i++) {
    		for(int j = 0; j < layers[i].neurons.length;j++) {
    			layers[i].neurons[j].update_weight();
    		}
    	}
    }
    
    public static float sumGradient(int n_index,int l_index) {
    	float gradient_sum = 0;
    	Layer current_layer = layers[l_index];
    	for(int i = 0; i < current_layer.neurons.length; i++) {
    		Neuron current_neuron = current_layer.neurons[i];
    		gradient_sum += current_neuron.weights[n_index]*current_neuron.gradient;
    	}
    	return gradient_sum;
    }
 
    public static void train(int training_iterations,float learning_rate) {
    	for(int i = 0; i < training_iterations; i++) {
    		for(int j = 0; j < tDataSet.length; j++) {
    			forward(tDataSet[j].data);
    			backward(learning_rate,tDataSet[j]);
    		}
    	}
    }
}