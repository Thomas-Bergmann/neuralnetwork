package de.hatoka.basicneuralnetwork;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.AfterEach;

import de.hatoka.basicneuralnetwork.activationfunctions.ActivationFunctions;

/**
 * Tests for parallel training functionality
 */
public class ParallelTrainingTest {
    
    private NeuralNetwork parallelNetwork;
    private NeuralNetwork sequentialNetwork;
    
    @AfterEach
    void cleanup() {
        if (parallelNetwork != null) {
            parallelNetwork.close();
        }
        if (sequentialNetwork != null) {
            sequentialNetwork.close();
        }
    }
    
    @Test
    void testParallelAndSequentialTrainingProduceSameResults() {
        // Create identical networks with same seed - one parallel, one sequential
        long seed = 12345L;
        
        parallelNetwork = NetworkBuilder.create(2, 1)
            .setHiddenLayers(1, 4)
            .setSeed(seed)
            .setLearningRate(0.1)
            .enableParallelTraining(2)
            .build();
            
        sequentialNetwork = NetworkBuilder.create(2, 1)
            .setHiddenLayers(1, 4)
            .setSeed(seed)
            .setLearningRate(0.1)
            .disableParallelTraining()
            .build();
        
        // Test data for XOR problem
        double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] targets = {{0}, {1}, {1}, {0}};
        
        // Train both networks with the same data
        for (int epoch = 0; epoch < 100; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                parallelNetwork.train(inputs[i], targets[i]);
                sequentialNetwork.train(inputs[i], targets[i]);
            }
        }
        
        // Test that both networks produce very similar results
        for (int i = 0; i < inputs.length; i++) {
            double[] parallelOutput = parallelNetwork.guess(inputs[i]);
            double[] sequentialOutput = sequentialNetwork.guess(inputs[i]);
            
            assertEquals(parallelOutput.length, sequentialOutput.length);
            for (int j = 0; j < parallelOutput.length; j++) {
                assertEquals(parallelOutput[j], sequentialOutput[j], 0.001, 
                    "Parallel and sequential training should produce similar results");
            }
        }
    }
    
    @Test
    void testParallelConfigurationSettings() {
        // Test default parallel configuration
        NeuralNetwork network = NetworkBuilder.create(2, 1)
            .enableParallelTraining()
            .build();
        
        assertTrue(network.isParallelTraining());
        assertEquals(Runtime.getRuntime().availableProcessors(), network.getParallelThreads());
        network.close();
        
        // Test custom thread count
        network = NetworkBuilder.create(2, 1)
            .enableParallelTraining(4)
            .build();
        
        assertTrue(network.isParallelTraining());
        assertEquals(4, network.getParallelThreads());
        network.close();
        
        // Test disabled parallel training
        network = NetworkBuilder.create(2, 1)
            .enableParallelTraining()
            .disableParallelTraining()
            .build();
        
        assertFalse(network.isParallelTraining());
        network.close();
    }
    
    @Test
    void testInvalidThreadCount() {
        assertThrows(IllegalArgumentException.class, () -> {
            NetworkBuilder.create(2, 1).enableParallelTraining(0);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            NetworkBuilder.create(2, 1).enableParallelTraining(-1);
        });
    }
    
    @Test
    void testParallelTrainingWithDifferentActivationFunctions() {
        long seed = 54321L;
        
        // Test with different activation functions
        ActivationFunctions[] functions = {ActivationFunctions.SIGMOID, ActivationFunctions.TANH, ActivationFunctions.RELU};
        
        for (ActivationFunctions function : functions) {
            parallelNetwork = NetworkBuilder.create(3, 2)
                .setHiddenLayers(2, 5)
                .setActivationFunction(function)
                .setSeed(seed)
                .enableParallelTraining(2)
                .build();
                
            sequentialNetwork = NetworkBuilder.create(3, 2)
                .setHiddenLayers(2, 5)
                .setActivationFunction(function)
                .setSeed(seed)
                .disableParallelTraining()
                .build();
            
            // Simple training data
            double[] input = {0.5, -0.2, 0.8};
            double[] target = {0.3, 0.7};
            
            // Train a few iterations
            for (int i = 0; i < 10; i++) {
                parallelNetwork.train(input, target);
                sequentialNetwork.train(input, target);
            }
            
            // Verify outputs are similar
            double[] parallelOutput = parallelNetwork.guess(input);
            double[] sequentialOutput = sequentialNetwork.guess(input);
            
            for (int j = 0; j < parallelOutput.length; j++) {
                assertEquals(parallelOutput[j], sequentialOutput[j], 0.001, 
                    "Results should be similar for " + function + " activation function");
            }
            
            parallelNetwork.close();
            sequentialNetwork.close();
            parallelNetwork = null;
            sequentialNetwork = null;
        }
    }
    
    @Test
    void testParallelPerformanceWithLargeNetwork() {
        // Test with a larger network to see if parallel operations provide benefit
        long seed = 98765L;
        
        NeuralNetwork largeParallelNetwork = NetworkBuilder.create(100, 50)
            .setHiddenLayers(3, 200)
            .setSeed(seed)
            .enableParallelTraining()
            .build();
        
        NeuralNetwork largeSequentialNetwork = NetworkBuilder.create(100, 50)
            .setHiddenLayers(3, 200)
            .setSeed(seed)
            .disableParallelTraining()
            .build();
        
        // Create larger input/output data
        double[] largeInput = new double[100];
        double[] largeTarget = new double[50];
        for (int i = 0; i < largeInput.length; i++) {
            largeInput[i] = Math.random() * 2 - 1; // Random values between -1 and 1
        }
        for (int i = 0; i < largeTarget.length; i++) {
            largeTarget[i] = Math.random() * 2 - 1;
        }
        
        // Time parallel training
        long parallelStart = System.nanoTime();
        for (int i = 0; i < 10; i++) {
            largeParallelNetwork.train(largeInput, largeTarget);
        }
        long parallelTime = System.nanoTime() - parallelStart;
        
        // Time sequential training
        long sequentialStart = System.nanoTime();
        for (int i = 0; i < 10; i++) {
            largeSequentialNetwork.train(largeInput, largeTarget);
        }
        long sequentialTime = System.nanoTime() - sequentialStart;
        
        // Verify both produce valid outputs (don't assert performance since it depends on hardware)
        double[] parallelOutput = largeParallelNetwork.guess(largeInput);
        double[] sequentialOutput = largeSequentialNetwork.guess(largeInput);
        
        assertEquals(parallelOutput.length, sequentialOutput.length);
        assertNotNull(parallelOutput);
        assertNotNull(sequentialOutput);
        
        System.out.printf("Large network training times - Parallel: %d ns, Sequential: %d ns%n", 
                         parallelTime, sequentialTime);
        
        largeParallelNetwork.close();
        largeSequentialNetwork.close();
    }
}