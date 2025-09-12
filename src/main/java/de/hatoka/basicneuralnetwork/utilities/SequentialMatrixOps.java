package de.hatoka.basicneuralnetwork.utilities;

import org.ejml.simple.SimpleMatrix;

/**
 * Sequential (non-parallel) implementation of matrix operations.
 * Uses standard EJML operations directly without any parallelization.
 */
public class SequentialMatrixOps implements MatrixOps {
    
    /**
     * Singleton instance for efficiency
     */
    public static final SequentialMatrixOps INSTANCE = new SequentialMatrixOps();
    
    /**
     * Private constructor to enforce singleton pattern
     */
    private SequentialMatrixOps() {
    }
    
    @Override
    public SimpleMatrix mult(SimpleMatrix a, SimpleMatrix b) {
        return a.mult(b);
    }
    
    @Override
    public SimpleMatrix plus(SimpleMatrix a, SimpleMatrix b) {
        return a.plus(b);
    }
    
    @Override
    public SimpleMatrix minus(SimpleMatrix a, SimpleMatrix b) {
        return a.minus(b);
    }
    
    @Override
    public SimpleMatrix elementMult(SimpleMatrix a, SimpleMatrix b) {
        return a.elementMult(b);
    }
    
    @Override
    public SimpleMatrix applyFunction(SimpleMatrix input, java.util.function.DoubleUnaryOperator function) {
        SimpleMatrix result = new SimpleMatrix(input.numRows(), input.numCols());
        for (int i = 0; i < input.getNumElements(); i++) {
            result.set(i, function.applyAsDouble(input.get(i)));
        }
        return result;
    }
}