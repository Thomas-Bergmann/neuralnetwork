package de.hatoka.basicneuralnetwork;

import java.util.Arrays;
import java.util.Objects;

import com.google.gson.annotations.Expose;

import de.hatoka.basicneuralnetwork.activationfunctions.ActivationFunctions;

public class NetworkConfiguration
{
    @Expose(serialize = true, deserialize = true)
    private final int inputNodes;
    @Expose(serialize = true, deserialize = true)
    private final int outputNodes;
    @Expose(serialize = true, deserialize = true)
    private final int[] hiddenLayers;
    @Expose(serialize = true, deserialize = true)
    private final double learningRate;
    @Expose(serialize = true, deserialize = true)
    private final ActivationFunctions activationFunction;
    @Expose(serialize = true, deserialize = true)
    private final long seed;

    NetworkConfiguration(int inputNodes, int outputNodes, int[] hiddenLayers, double learningRate,
                    ActivationFunctions activationFunction, long seed)
    {
        this.inputNodes = inputNodes;
        this.outputNodes = outputNodes;
        this.hiddenLayers = hiddenLayers;
        this.learningRate = learningRate;
        this.activationFunction = activationFunction;
        this.seed = seed;
    }

    public int getInputNodes()
    {
        return inputNodes;
    }

    public int getOutputNodes()
    {
        return outputNodes;
    }

    public int[] getHiddenLayers()
    {
        return hiddenLayers;
    }

    public double getLearningRate()
    {
        return learningRate;
    }

    public ActivationFunctions getActivationFunction()
    {
        return activationFunction;
    }

    public long getSeed()
    {
        return seed;
    }

    @Override
    public int hashCode()
    {
        final int prime = 31;
        int result = 1;
        result = prime * result + Arrays.hashCode(hiddenLayers);
        result = prime * result + Objects.hash(activationFunction, inputNodes, learningRate, outputNodes, seed);
        return result;
    }

    @Override
    public boolean equals(Object obj)
    {
        if (this == obj) return true;
        if (obj == null) return false;
        if (obj instanceof NetworkConfiguration other)
        {
            return activationFunction == other.activationFunction && Arrays.equals(hiddenLayers, other.hiddenLayers)
                            && inputNodes == other.inputNodes
                            && Double.doubleToLongBits(learningRate) == Double.doubleToLongBits(other.learningRate)
                            && outputNodes == other.outputNodes && seed == other.seed;
        }
        return false;
    }

}
