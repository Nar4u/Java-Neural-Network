public class Layer {
	public Neuron[] neurons;


	//Setting up the Layers
	public Layer(int inNeurons,int numberNeurons) {
		this.neurons = new Neuron[numberNeurons];
		
		for(int i = 0; i < numberNeurons; i++) {
			float[] weights = new float[inNeurons];
			for(int j = 0; j < inNeurons; j++) {
				weights[j] = StatUtil.RandomFloat(Neuron.minWeightValue, Neuron.maxWeightValue);
			}
			neurons[i] = new Neuron(weights,StatUtil.RandomFloat(-1, 1));
		}
	}

	public Layer(float input[]) {
		this.neurons = new Neuron[input.length];
		for(int i = 0; i < input.length; i++) {
			this.neurons[i] = new Neuron(input[i]);
		}
	}
}
