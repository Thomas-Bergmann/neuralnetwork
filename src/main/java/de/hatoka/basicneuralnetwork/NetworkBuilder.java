package de.hatoka.basicneuralnetwork;

import java.util.Random;

import de.hatoka.basicneuralnetwork.activationfunctions.ActivationFunctions;

public class NetworkBuilder
{
    /**
     * randomizer to create a random seed for the random of the network (can be used for testing)
     */
    private static final Random initRandom = new Random();

    public static NetworkBuilder create(int inputNodes, int outputNodes)
    {
        return new NetworkBuilder(new NetworkConfiguration(inputNodes, outputNodes, new int[] {}, 0.1,
                        ActivationFunctions.SIGMOID, initRandom.nextLong()));
    }

    private final NetworkConfiguration config;

    private NetworkBuilder(NetworkConfiguration config)
    {
        this.config = config;
    }

    public NeuralNetwork build()
    {
        return new NeuralNetwork(config);
    }

    public NetworkBuilder setHiddenLayers(int hiddenLayers, int hiddenNodes)
    {
        int[] hiddenLayerDefinition = new int[hiddenLayers];
        for (int i = 0; i < hiddenLayers; i++)
        {
            hiddenLayerDefinition[i] = hiddenNodes;
        }
        NetworkConfiguration newConfig = new NetworkConfiguration(config.getInputNodes(), config.getOutputNodes(),
                        hiddenLayerDefinition, config.getLearningRate(), config.getActivationFunction(),
                        config.getSeed());
        return new NetworkBuilder(newConfig);
    }

    public NetworkBuilder setActivationFunction(ActivationFunctions activationFunction)
    {
        NetworkConfiguration newConfig = new NetworkConfiguration(config.getInputNodes(), config.getOutputNodes(),
                        config.getHiddenLayers(), config.getLearningRate(), activationFunction, config.getSeed());
        return new NetworkBuilder(newConfig);
    }

    public NetworkBuilder setSeed(long seed)
    {
        NetworkConfiguration newConfig = new NetworkConfiguration(config.getInputNodes(), config.getOutputNodes(),
                        config.getHiddenLayers(), config.getLearningRate(), config.getActivationFunction(), seed);
        return new NetworkBuilder(newConfig);
    }
}
