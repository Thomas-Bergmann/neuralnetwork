package de.hatoka.basicneuralnetwork;

import java.util.Arrays;
import java.util.Objects;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import com.google.gson.annotations.Expose;

import de.hatoka.basicneuralnetwork.activationfunctions.ActivationFunctions;
import de.hatoka.basicneuralnetwork.utilities.MatrixUtilities;

/**
 * Created by KimFeichtinger on 04.03.18.
 */
public class NeuralNetwork
{
    /**
     * randomizer to initialize the network
     */
    private Random random;

    /**
     * Configuration of network
     */
    @Expose(serialize = true, deserialize = true)
    private final NetworkConfiguration config;

    @Expose(serialize = true, deserialize = true)
    private SimpleMatrix[] weights;
    @Expose(serialize = true, deserialize = true)
    private SimpleMatrix[] biases;

    /**
     * Generate a new neural network without hidden layers
     * @param inputNodes number of input nodes
     * @param outputNodes number of output nodes
     * @return generated network
     */
    public static NeuralNetwork build(int inputNodes, int outputNodes)
    {
        return NetworkBuilder.create(inputNodes, outputNodes).build();
    }

    /**
     * Generate a new neural network with 1 hidden layer with the given amount of nodes
     * @param inputNodes number of input nodes
     * @param hiddenNodes number of hidden nodes
     * @param outputNodes number of output nodes
     * @return generated network
     */
    public static NeuralNetwork build(int inputNodes, int hiddenNodes, int outputNodes)
    {
        return NetworkBuilder.create(inputNodes, outputNodes).setHiddenLayers(1, hiddenNodes).build();
    }

    /**
     * Generate a new neural network with multiple hidden layers with same amount of nodes per hidden layer
     * @param inputNodes number of input nodes
     * @param hiddenLayers number of hidden layers
     * @param hiddenNodes number of hidden nodes per hidden layer
     * @param outputNodes number of output nodes
     * @return generated network
     */
    public static NeuralNetwork build(int inputNodes, int hiddenLayers, int hiddenNodes, int outputNodes)
    {
        return NetworkBuilder.create(inputNodes, outputNodes).setHiddenLayers(hiddenLayers, hiddenNodes).build();
    }

    /**
     * Constructor a new neural network with multiple hidden layers with same amount of nodes per hidden layer
     * @param config configuration of network
     */
    NeuralNetwork(NetworkConfiguration config)
    {
        this.config = config;
        this.random = new Random(config.getSeed());
        initializeWeights();
        initializeBiases();
    }

    /**
     * Initializes the network after loading
     */
    public void afterLoad()
    {
        this.random = new Random(config.getSeed());
    }
    /**
     * Constructor to copy an existing network
     * @param nn source network
     */
    private NeuralNetwork(NeuralNetwork nn)
    {
        this.config = nn.config;
        this.random = nn.random;

        this.weights = new SimpleMatrix[config.getHiddenLayers().length + 1];
        for (int i = 0; i < nn.weights.length; i++)
        {
            this.weights[i] = nn.weights[i].copy();
        }

        this.biases = new SimpleMatrix[config.getHiddenLayers().length + 1];
        for (int i = 0; i < nn.biases.length; i++)
        {
            this.biases[i] = nn.biases[i].copy();
        }
    }

    /**
     * @param numRows number of rows
     * @param numCols number of columns
     * @return a matrix with given dimensions with random values from -1 to 1
     */
    private SimpleMatrix randomMatrix(int numRows, int numCols)
    {
        return SimpleMatrix.random64(numRows, numCols, -1, 1, random);
    }

    /**
     * Initialize weights with random numbers between -1 and 1
     */
    private void initializeWeights()
    {
        int hiddenLayers = config.getHiddenLayers().length;
        weights = new SimpleMatrix[hiddenLayers + 1];

        // 1st weights that connects inputs to first hidden nodes or output nodes if no hidden nodes exist
        if (hiddenLayers == 0)
        {
            weights[0] = randomMatrix(config.getOutputNodes(), config.getInputNodes());
            return;
        }

        weights[0] = randomMatrix(config.getHiddenLayers()[0], config.getInputNodes());
        // Initialize the weights between the layers and fill them with random values
        for (int i = 1; i < hiddenLayers; i++)
        {
            weights[i] = randomMatrix(config.getHiddenLayers()[i], config.getHiddenLayers()[i-1]);
        }
        // last weights that connect last hidden layer to output
        weights[hiddenLayers] = randomMatrix(config.getOutputNodes(), config.getHiddenLayers()[hiddenLayers - 1]);
    }

    /**
     * Each hidden layer and the output layer gets a biases as additional input
     */
    private void initializeBiases()
    {
        int hiddenLayers = config.getHiddenLayers().length;
        biases = new SimpleMatrix[hiddenLayers + 1];

        // Initialize the biases and fill them with random values
        for (int i = 0; i < hiddenLayers; i++)
        {
            biases[i] = randomMatrix(config.getHiddenLayers()[i], 1);
        }
        biases[hiddenLayers] = randomMatrix(config.getOutputNodes(), 1);
    }

    /**
     * @param input list of input values for the network
     * @return list of output values calculated (guess) by the network via forward propagation
     */
    public double[] guess(double[] input)
    {
        if (input.length != config.getInputNodes())
        {
            throw new WrongDimensionException(input.length, config.getInputNodes(), "Input");
        }
        // Transform array to matrix
        SimpleMatrix output = MatrixUtilities.arrayToMatrix(input);
        for (int i = 0; i < config.getHiddenLayers().length + 1; i++)
        {
            output = calculateLayer(weights[i], biases[i], output);
        }
        return MatrixUtilities.getColumnFromMatrixAsArray(output, 0);
    }

    public void train(double[] inputArray, double[] targetArray)
    {
        if (inputArray.length != config.getInputNodes())
        {
            throw new WrongDimensionException(inputArray.length, config.getInputNodes(), "Input");
        }
        else if (targetArray.length != config.getOutputNodes())
        {
            throw new WrongDimensionException(targetArray.length, config.getOutputNodes(), "Output");
        }
        else
        {
            // Transform 2D array to matrix
            SimpleMatrix input = MatrixUtilities.arrayToMatrix(inputArray);
            SimpleMatrix target = MatrixUtilities.arrayToMatrix(targetArray);

            // Calculate the values of every single layer
            SimpleMatrix layers[] = new SimpleMatrix[config.getHiddenLayers().length + 2];
            layers[0] = input;
            for (int j = 0; j < config.getHiddenLayers().length + 1; j++)
            {
                input = layers[j+1] = calculateLayer(weights[j], biases[j], input);
            }

            for (int n = config.getHiddenLayers().length + 1; n > 0; n--)
            {
                // Calculate error
                SimpleMatrix errors = target.minus(layers[n]);

                // Calculate gradient
                SimpleMatrix gradients = calculateGradient(layers[n], errors);

                // Calculate delta
                SimpleMatrix deltas = calculateDeltas(gradients, layers[n - 1]);

                // Apply gradient to bias
                biases[n - 1] = biases[n - 1].plus(gradients);

                // Apply delta to weights
                weights[n - 1] = weights[n - 1].plus(deltas);

                // Calculate and set target for previous (next) layer
                SimpleMatrix previousError = weights[n - 1].transpose().mult(errors);
                target = previousError.plus(layers[n - 1]);
            }
        }
    }

    // Generates an exact copy of a NeuralNetwork
    public NeuralNetwork copy()
    {
        return new NeuralNetwork(this);
    }

    /**
     * Merges the weights and biases of two NeuralNetworks and returns a new object
     * Merge-ratio: 50:50 (half of the values will be from nn1 and other half from nn2)
     * @param nn network to merge with current
     * @return merged network
     */
    public NeuralNetwork merge(NeuralNetwork nn)
    {
        return this.merge(nn, 0.5);
    }

    /**
     * Merges the weights and biases of two NeuralNetworks and returns a new object
     * Merge-ratio: 50:50 (half of the values will be from nn1 and other half from nn2)
     * @param nn network to merge with current
     * @param probability ratio between current and other network
     * @return merged network
     */
    public NeuralNetwork merge(NeuralNetwork nn, double probability)
    {
        // Check whether the nns have the same dimensions
        if (!Arrays.equals(this.getDimensions(), nn.getDimensions()))
        {
            throw new WrongDimensionException(this.getDimensions(), nn.getDimensions());
        }
        else
        {
            NeuralNetwork result = this.copy();

            for (int i = 0; i < result.weights.length; i++)
            {
                result.weights[i] = MatrixUtilities.mergeMatrices(this.weights[i], nn.weights[i], probability);
            }

            for (int i = 0; i < result.biases.length; i++)
            {
                result.biases[i] = MatrixUtilities.mergeMatrices(this.biases[i], nn.biases[i], probability);
            }
            return result;
        }
    }

    // Gaussian mutation with given probability, Slightly modifies values (weights +
    // biases) with given probability
    // Probability: number between 0 and 1
    // Depending on probability more/ less values will be mutated (e.g. prob = 1.0:
    // all the values will be mutated)
    public void mutate(double probability)
    {
        applyMutation(weights, probability);
        applyMutation(biases, probability);
    }

    // Adds a randomly generated gaussian number to each element of a Matrix in an
    // array of matrices
    // Probability: determines how many values will be modified
    private void applyMutation(SimpleMatrix[] matrices, double probability)
    {
        for (SimpleMatrix matrix : matrices)
        {
            for (int j = 0; j < matrix.getNumElements(); j++)
            {
                if (random.nextDouble() < probability)
                {
                    double offset = random.nextGaussian() / 2;
                    matrix.set(j, matrix.get(j) + offset);
                }
            }
        }
    }

    // Generic function to calculate one layer
    private SimpleMatrix calculateLayer(SimpleMatrix weights, SimpleMatrix bias, SimpleMatrix input)
    {
        // Calculate outputs of layer
        SimpleMatrix result = weights.mult(input);
        // Add bias to outputs
        result = result.plus(bias);
        // Apply activation function and return result
        return applyActivationFunction(result, false);
    }

    private SimpleMatrix calculateGradient(SimpleMatrix layer, SimpleMatrix error)
    {
        SimpleMatrix gradient = applyActivationFunction(layer, true);
        gradient = gradient.elementMult(error);
        return gradient.scale(config.getLearningRate());
    }

    private SimpleMatrix calculateDeltas(SimpleMatrix gradient, SimpleMatrix layer)
    {
        return gradient.mult(layer.transpose());
    }

    // Applies an activation function to a matrix
    // An object of an implementation of the ActivationFunction-interface has to be
    // passed
    // The function in this class will be to the matrix
    private SimpleMatrix applyActivationFunction(SimpleMatrix input, boolean derivative)
    {
        // Applies either derivative of activation function or regular activation
        // function to a matrix and returns the result
        return derivative ? config.getActivationFunction().getFunction().invert(input)
                        : config.getActivationFunction().getFunction().activate(input);
    }

    public ActivationFunctions getActivationFunction()
    {
        return config.getActivationFunction();
    }

    public double getLearningRate()
    {
        return config.getLearningRate();
    }

    public int getInputNodes()
    {
        return config.getInputNodes();
    }

    public int getHiddenLayers()
    {
        return config.getHiddenLayers().length;
    }

    public int getHiddenNodes()
    {
        return config.getHiddenLayers().length == 0 ? 0 : config.getHiddenLayers()[0];
    }

    public int getOutputNodes()
    {
        return config.getOutputNodes();
    }

    public SimpleMatrix[] getWeights()
    {
        return weights;
    }

    public void setWeights(SimpleMatrix[] weights)
    {
        this.weights = weights;
    }

    public SimpleMatrix[] getBiases()
    {
        return biases;
    }

    public void setBiases(SimpleMatrix[] biases)
    {
        this.biases = biases;
    }

    public int[] getDimensions()
    {
        // TODO doesn't fit to networks with different hidden layers
        return new int[] { config.getInputNodes(), config.getHiddenLayers().length, config.getHiddenLayers()[0], config.getOutputNodes() };
    }

    private int hashCode(SimpleMatrix[] matrices)
    {
        final int prime = 31;
        int result = 1;
        for (int i = 0; i < matrices.length; i++)
        {
            result = prime * result + hashCode(matrices[i]);
        }
        return result;
    }

    private int hashCode(SimpleMatrix matrix)
    {
        final int prime = 31;
        int result = 1;
        for (int i = 0; i < matrix.getNumElements(); i++)
        {
            result = prime * result + Double.valueOf(matrix.get(i)).hashCode();
        }
        return result;
    }

    private boolean equalsMatrix(SimpleMatrix[] a, SimpleMatrix[] b)
    {
        if (a.length != b.length)
        {
            return false;
        }
        for (int i = 0; i < a.length; i++)
        {
            if (!equalsMatrix(a[i], b[i]))
            {
                return false;
            }
        }
        return true;
    }

    private boolean equalsMatrix(SimpleMatrix a, SimpleMatrix b)
    {
        return a.isIdentical(b, 0d);
    }

    @Override
    public int hashCode()
    {
        final int prime = 31;
        int result = 1;
        result = prime * result + Objects.hash(config);
        result = prime * result + hashCode(biases);
        result = prime * result + hashCode(weights);
        return result;
    }

    @Override
    public boolean equals(Object obj)
    {
        if (this == obj) return true;
        if (obj == null) return false;
        if (getClass() != obj.getClass()) return false;
        NeuralNetwork other = (NeuralNetwork)obj;
        return Objects.equals(config, other.config) && equalsMatrix(biases, other.biases) && equalsMatrix(weights, other.weights);
    }
}
