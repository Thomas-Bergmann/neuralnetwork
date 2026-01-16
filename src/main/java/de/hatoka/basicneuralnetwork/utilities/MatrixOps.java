package de.hatoka.basicneuralnetwork.utilities;

import org.ejml.simple.SimpleMatrix;

/**
 * Interface for matrix operations used in neural network training.
 * Provides strategy pattern for switching between sequential and parallel implementations.
 */
public interface MatrixOps extends AutoCloseable {
    
    /**
     * Matrix multiplication
     * @param a first matrix
     * @param b second matrix
     * @return result of a * b
     */
    SimpleMatrix mult(SimpleMatrix a, SimpleMatrix b);
    
    /**
     * Matrix addition
     * @param a first matrix
     * @param b second matrix
     * @return result of a + b
     */
    SimpleMatrix plus(SimpleMatrix a, SimpleMatrix b);
    
    /**
     * Matrix subtraction
     * @param a first matrix
     * @param b second matrix
     * @return result of a - b
     */
    SimpleMatrix minus(SimpleMatrix a, SimpleMatrix b);
    
    /**
     * Element-wise matrix multiplication
     * @param a first matrix
     * @param b second matrix
     * @return result of element-wise a * b
     */
    SimpleMatrix elementMult(SimpleMatrix a, SimpleMatrix b);
    
    /**
     * Apply a function to each element of the matrix
     * @param input input matrix
     * @param function function to apply
     * @return new matrix with function applied to each element
     */
    SimpleMatrix applyFunction(SimpleMatrix input, java.util.function.DoubleUnaryOperator function);
    
    /**
     * Cleanup any resources used by this MatrixOps implementation
     */
    @Override
    default void close() {
        // Default implementation does nothing
    }
}