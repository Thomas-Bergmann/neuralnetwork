package de.hatoka.basicneuralnetwork;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;

import de.hatoka.basicneuralnetwork.activationfunctions.ActivationFunctions;
import de.hatoka.basicneuralnetwork.utilities.FileReaderAndWriter;

class NeuralNetworkTest
{
    private static final int inputNodes = 1;
    private static final int hiddenLayers = 2;
    private static final int hiddenNodes = 3;
    private static final int outputNodes = 4;
    private static final double ADAPTION_THRESHOLD = 1E-2;
    private static final NetworkBuilder DEFAULT_BUILDER = NetworkBuilder.create(inputNodes, outputNodes).setHiddenLayers(hiddenLayers, hiddenNodes);

    /**
     * Default network for tests
     */
    private NeuralNetwork nn;

    @BeforeEach
    public void setup()
    {
        nn = DEFAULT_BUILDER.build();
    }

    @Test
    public void initializeDefaultValuesTest()
    {
        assertEquals(0.1, nn.getLearningRate());
        assertEquals(ActivationFunctions.SIGMOID, nn.getActivationFunction());
    }

    @Test
    public void initializeWeightsTest()
    {
        SimpleMatrix[] weights = nn.getWeights();
        assertEquals(hiddenLayers + 1, weights.length);
        assertEquals(inputNodes, weights[0].numCols());
        assertEquals(hiddenNodes, weights[0].numRows());
        assertEquals(hiddenNodes, weights[weights.length - 1].numCols());
        assertEquals(outputNodes, weights[weights.length - 1].numRows());

        for (SimpleMatrix weight : weights)
        {
            for (int i = 0; i < weight.getNumElements(); i++)
            {
                double value = weight.get(i);
                assertTrue(value >= -1 && value <= 1);
            }
        }
    }

    @Test
    public void initializeBiasesTest()
    {
        SimpleMatrix[] biases = nn.getBiases();
        assertEquals(hiddenLayers + 1, biases.length);
        assertEquals(1, biases[0].numCols());
        assertEquals(hiddenNodes, biases[0].numRows());
        assertEquals(1, biases[biases.length - 1].numCols());
        assertEquals(outputNodes, biases[biases.length - 1].numRows());

        for (SimpleMatrix bias : biases)
        {
            for (int i = 0; i < bias.getNumElements(); i++)
            {
                double value = bias.get(i);
                assertTrue(value >= -1 && value <= 1);
            }
        }
    }

    @Test
    public void guessTestWrongDimension()
    {
        Throwable exception = assertThrows(WrongDimensionException.class, () -> nn.guess(new double[] { 0, 1 }));
        assertEquals("Expected 1 value(s) for Input-layer but got 2.", exception.getMessage());
    }

    @Test
    public void guessTest()
    {
        double[] result = nn.guess(new double[] { 1 });
        assertEquals(outputNodes, result.length);
    }

    @Test
    public void trainTestWrongDimensionInput()
    {
        Throwable exception = assertThrows(WrongDimensionException.class,
                        () -> nn.train(new double[] { 0, 1 }, new double[] { 0, 1 }));
        assertEquals("Expected 1 value(s) for Input-layer but got 2.", exception.getMessage());
    }

    @Test
    public void trainTestWrongDimensionOutput()
    {
        Throwable exception = assertThrows(WrongDimensionException.class,
                        () -> nn.train(new double[] { 0 }, new double[] { 0, 1 }));
        assertEquals("Expected 4 value(s) for Output-layer but got 2.", exception.getMessage());
    }

    @Test
    public void trainTest()
    {
        assertAll(() -> nn.train(new double[] { 0 }, new double[] { 0, 1, 2, 3 }));
    }

    @Test
    public void equalsTest()
    {
        NeuralNetwork nnB = nn.copy();
        assertEquals(nnB.hashCode(), nn.hashCode());
        assertEquals(nnB, nn);
    }

    @Test
    public void copyTest()
    {
        NeuralNetwork nnB = nn.copy();

        assertEquals(nn.getInputNodes(), nnB.getInputNodes());
        assertEquals(nn.getHiddenNodes(), nnB.getHiddenNodes());
        assertEquals(nn.getHiddenLayers(), nnB.getHiddenLayers());
        assertEquals(nn.getOutputNodes(), nnB.getOutputNodes());
        assertEquals(nn.getLearningRate(), nnB.getLearningRate());
        assertEquals(nn.getActivationFunction(), nnB.getActivationFunction());

        assertEquals(nn, nnB);
    }

    @Test
    public void mergeTestWrongDimension()
    {
        NeuralNetwork nnM = NetworkBuilder.create(2, 5).setHiddenLayers(3, 4).build();
        Throwable exception = assertThrows(WrongDimensionException.class, () -> nn.merge(nnM));
        assertEquals("The dimensions of these two neural networks don't match: [1, 2, 3, 4], [2, 3, 4, 5]",
                        exception.getMessage());
    }

    @Test
    public void mergeTest()
    {
        NeuralNetwork nnB = DEFAULT_BUILDER.build();
        NeuralNetwork result = nn.merge(nnB);

        assertNotEquals(nn, result);
        assertNotEquals(nnB, result);

        assertEquals(nn.getLearningRate(), result.getLearningRate());
        assertEquals(nn.getActivationFunction(), result.getActivationFunction());

        for (int i = 0; i < result.getWeights().length; i++)
        {
            SimpleMatrix resultWeights = result.getWeights()[i];
            SimpleMatrix nnWeights = nn.getWeights()[i];
            SimpleMatrix nnBWeights = nnB.getWeights()[i];
            for (int j = 0; j < resultWeights.getNumElements(); j++)
            {
                double value = resultWeights.get(j);
                assertTrue(value == nnWeights.get(j) || value == nnBWeights.get(j));
            }
        }

        for (int i = 0; i < result.getBiases().length; i++)
        {
            SimpleMatrix resultBiases = result.getBiases()[i];
            SimpleMatrix nnBiases = nn.getBiases()[i];
            SimpleMatrix nnBBiases = nnB.getBiases()[i];
            for (int j = 0; j < resultBiases.getNumElements(); j++)
            {
                double value = resultBiases.get(j);
                assertTrue(value == nnBiases.get(j) || value == nnBBiases.get(j));
            }
        }
    }

    @Test
    public void mutateTest()
    {
        assertAll(() -> nn.mutate(0.5));
    }

    @Test
    public void testConfigurationChanges()
    {
        nn = DEFAULT_BUILDER.setActivationFunction(ActivationFunctions.TANH).setLearningRate(0.4).build();
        assertEquals(ActivationFunctions.TANH, nn.getActivationFunction());
        assertEquals(0.4, nn.getLearningRate());
    }

    @Test
    public void testNetworkWithNoHiddenNodes()
    {
        NeuralNetwork noHiddenNodesNN = NetworkBuilder.create(3, 2).build();
        assertEquals(0, noHiddenNodesNN.getHiddenNodes());
        assertEquals(0, noHiddenNodesNN.getHiddenLayers());
        // one matrix for layer between input and output
        assertEquals(1, noHiddenNodesNN.getWeights().length);
        SimpleMatrix matrix = noHiddenNodesNN.getWeights()[0];
        assertEquals(6, matrix.getNumElements());
        for (int element = 0; element < matrix.getNumElements(); element++)
        {
            assertNotEquals(0, matrix.get(element));
        }
    }

    @Test
    public void testOr()
    {
        NeuralNetwork nn = NetworkBuilder.create(2, 1).build();
        for (int i = 0; i < 2_000; i++)
        {
            double adaption = 0;
            adaption += nn.train(asArray(0, 0), asArray(0));
            adaption += nn.train(asArray(0, 1), asArray(1));
            adaption += nn.train(asArray(1, 0), asArray(1));
            adaption += nn.train(asArray(1, 1), asArray(1));
            if (adaption < ADAPTION_THRESHOLD)
            {
                LoggerFactory.getLogger(getClass())
                             .info("network learning finished after {} iterations adaption was {}.", i, adaption);
                break;
            }
            if (i % 100 == 0) LoggerFactory.getLogger(getClass()).trace("network adaption {}.", adaption);
        }
        assertGuessFalse(nn.guess(asArray(0, 0))[0]);
        assertGuessTrue(nn.guess(asArray(0, 1))[0]);
        assertGuessTrue(nn.guess(asArray(1, 0))[0]);
        assertGuessTrue(nn.guess(asArray(1, 1))[0]);
    }

    @Test
    public void testXOr()
    {
        NeuralNetwork nn = NetworkBuilder.create(2, 1).setHiddenLayers(1, 4).build();
        // around 8000 iterations are necessary
        for (int i = 0; i < 10_000; i++)
        {
            double adaption = 0;
            adaption += nn.train(asArray(0, 0), asArray(0));
            adaption += nn.train(asArray(0, 1), asArray(1));
            adaption += nn.train(asArray(1, 0), asArray(1));
            adaption += nn.train(asArray(1, 1), asArray(0));
            if (adaption < ADAPTION_THRESHOLD)
            {
                LoggerFactory.getLogger(getClass())
                             .info("network learning finished after {} iterations adaption was {}.", i, adaption);
                break;
            }
            if (i % 100 == 0) LoggerFactory.getLogger(getClass()).trace("network adaption {}.", adaption);
        }
        LoggerFactory.getLogger(getClass()).trace("network {}", new FileReaderAndWriter().asJson(nn));
        assertGuessFalse(nn.guess(asArray(0, 0))[0]);
        assertGuessTrue(nn.guess(asArray(0, 1))[0]);
        assertGuessTrue(nn.guess(asArray(1, 0))[0]);
        assertGuessFalse(nn.guess(asArray(1, 1))[0]);
    }

    @Test
    public void testNAndForBiasUse()
    {
        NeuralNetwork nn = NetworkBuilder.create(2, 1).build();
        // around 900 iterations are necessary
        for (int i = 0; i < 2_000; i++)
        {
            double adaption = 0;
            adaption += nn.train(asArray(0, 0), asArray(1));
            adaption += nn.train(asArray(0, 1), asArray(0));
            adaption += nn.train(asArray(1, 0), asArray(0));
            adaption += nn.train(asArray(1, 1), asArray(0));
            if (adaption < ADAPTION_THRESHOLD)
            {
                LoggerFactory.getLogger(getClass())
                             .info("network learning finished after {} iterations adaption was {}.", i, adaption);
                break;
            }
            if (i % 100 == 0) LoggerFactory.getLogger(getClass()).trace("network adaption {}.", adaption);
        }
        LoggerFactory.getLogger(getClass()).trace("network {}", new FileReaderAndWriter().asJson(nn));
        assertGuessTrue(nn.guess(asArray(0, 0))[0]);
        assertGuessFalse(nn.guess(asArray(0, 1))[0]);
        assertGuessFalse(nn.guess(asArray(1, 0))[0]);
        assertGuessFalse(nn.guess(asArray(1, 1))[0]);
    }

    private void assertGuessTrue(double value)
    {
        assertTrue(value > 0.7, "value should greater than 0.7, but is " + value);
    }

    private void assertGuessFalse(double value)
    {
        assertTrue(value < 0.3, "value should less than 0.3, but is " + value);
    }

    private double[] asArray(double... values)
    {
        return values;
    }
}
