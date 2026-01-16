package de.hatoka.basicneuralnetwork;

import java.util.Arrays;
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
                        ActivationFunctions.SIGMOID, 0, false, Runtime.getRuntime().availableProcessors()));
    }

    private final NetworkConfiguration config;

    private NetworkBuilder(NetworkConfiguration config)
    {
        this.config = config;
    }

    /**
     * @return the configured network. if the seed is not set this method will random initialize the network
     */
    public NeuralNetwork build()
    {
        if (config.getSeed() == 0L)
        {
            return setSeed(initRandom.nextLong()).build();
        }
        return new NeuralNetwork(config);
    }

    public NetworkBuilder setHiddenLayers(int hiddenLayers, int hiddenNodes)
    {
        int[] hiddenLayerDefinition = new int[hiddenLayers];
        Arrays.fill(hiddenLayerDefinition, hiddenNodes);
        NetworkConfiguration newConfig = new NetworkConfiguration(config.getInputNodes(), config.getOutputNodes(),
                        hiddenLayerDefinition, config.getLearningRate(), config.getActivationFunction(),
                        config.getSeed(), config.isParallelTraining(), config.getParallelThreads());
        return new NetworkBuilder(newConfig);
    }

    public NetworkBuilder setActivationFunction(ActivationFunctions activationFunction)
    {
        NetworkConfiguration newConfig = new NetworkConfiguration(config.getInputNodes(), config.getOutputNodes(),
                        config.getHiddenLayers(), config.getLearningRate(), activationFunction, config.getSeed(),
                        config.isParallelTraining(), config.getParallelThreads());
        return new NetworkBuilder(newConfig);
    }

    public NetworkBuilder setSeed(long seed)
    {
        NetworkConfiguration newConfig = new NetworkConfiguration(config.getInputNodes(), config.getOutputNodes(),
                        config.getHiddenLayers(), config.getLearningRate(), config.getActivationFunction(), seed,
                        config.isParallelTraining(), config.getParallelThreads());
        return new NetworkBuilder(newConfig);
    }

    public NetworkBuilder setLearningRate(double learningRate)
    {
        NetworkConfiguration newConfig = new NetworkConfiguration(config.getInputNodes(), config.getOutputNodes(),
                        config.getHiddenLayers(), learningRate, config.getActivationFunction(), config.getSeed(),
                        config.isParallelTraining(), config.getParallelThreads());
        return new NetworkBuilder(newConfig);
    }

    /**
     * Enable parallel training with default thread count (available processors)
     * @return NetworkBuilder with parallel training enabled
     */
    public NetworkBuilder enableParallelTraining()
    {
        return enableParallelTraining(Runtime.getRuntime().availableProcessors());
    }

    /**
     * Enable parallel training with specified thread count
     * @param threads number of threads to use for parallel operations
     * @return NetworkBuilder with parallel training enabled
     */
    public NetworkBuilder enableParallelTraining(int threads)
    {
        if (threads <= 0) {
            throw new IllegalArgumentException("Thread count must be positive");
        }
        NetworkConfiguration newConfig = new NetworkConfiguration(config.getInputNodes(), config.getOutputNodes(),
                        config.getHiddenLayers(), config.getLearningRate(), config.getActivationFunction(),
                        config.getSeed(), true, threads);
        return new NetworkBuilder(newConfig);
    }

    /**
     * Disable parallel training (use sequential operations)
     * @return NetworkBuilder with parallel training disabled
     */
    public NetworkBuilder disableParallelTraining()
    {
        NetworkConfiguration newConfig = new NetworkConfiguration(config.getInputNodes(), config.getOutputNodes(),
                        config.getHiddenLayers(), config.getLearningRate(), config.getActivationFunction(),
                        config.getSeed(), false, config.getParallelThreads());
        return new NetworkBuilder(newConfig);
    }
}
