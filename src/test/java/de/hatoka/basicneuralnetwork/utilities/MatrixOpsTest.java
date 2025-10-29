package de.hatoka.basicneuralnetwork.utilities;

import static org.junit.jupiter.api.Assertions.*;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;

/**
 * Tests to verify MatrixOps interface implementations work correctly
 */
public class MatrixOpsTest {
    
    @Test
    void testSequentialMatrixOps() {
        MatrixOps ops = SequentialMatrixOps.INSTANCE;
        testMatrixOperations(ops);
    }
    
    @Test
    void testParallelMatrixOps() {
        try (ParallelMatrixOps ops = new ParallelMatrixOps(2)) {
            testMatrixOperations(ops);
        }
    }
    
    @Test
    void testMatrixOpsEquivalence() {
        SimpleMatrix a = new SimpleMatrix(new double[][]{{1, 2}, {3, 4}});
        SimpleMatrix b = new SimpleMatrix(new double[][]{{5, 6}, {7, 8}});
        
        MatrixOps sequential = SequentialMatrixOps.INSTANCE;
        
        try (ParallelMatrixOps parallel = new ParallelMatrixOps(2)) {
            // Test that both implementations produce the same results
            SimpleMatrix seqMult = sequential.mult(a, b);
            SimpleMatrix parMult = parallel.mult(a, b);
            assertTrue(seqMult.isIdentical(parMult, 1e-10));
            
            SimpleMatrix seqPlus = sequential.plus(a, b);
            SimpleMatrix parPlus = parallel.plus(a, b);
            assertTrue(seqPlus.isIdentical(parPlus, 1e-10));
            
            SimpleMatrix seqMinus = sequential.minus(a, b);
            SimpleMatrix parMinus = parallel.minus(a, b);
            assertTrue(seqMinus.isIdentical(parMinus, 1e-10));
            
            SimpleMatrix seqElemMult = sequential.elementMult(a, b);
            SimpleMatrix parElemMult = parallel.elementMult(a, b);
            assertTrue(seqElemMult.isIdentical(parElemMult, 1e-10));
        }
    }
    
    private void testMatrixOperations(MatrixOps ops) {
        SimpleMatrix a = new SimpleMatrix(new double[][]{{1, 2}, {3, 4}});
        SimpleMatrix b = new SimpleMatrix(new double[][]{{5, 6}, {7, 8}});
        
        // Test matrix multiplication
        SimpleMatrix mult = ops.mult(a, b);
        assertEquals(2, mult.numRows());
        assertEquals(2, mult.numCols());
        assertEquals(19, mult.get(0, 0), 1e-10); // 1*5 + 2*7
        assertEquals(22, mult.get(0, 1), 1e-10); // 1*6 + 2*8
        
        // Test matrix addition
        SimpleMatrix plus = ops.plus(a, b);
        assertEquals(6, plus.get(0, 0), 1e-10); // 1 + 5
        assertEquals(8, plus.get(0, 1), 1e-10); // 2 + 6
        
        // Test matrix subtraction
        SimpleMatrix minus = ops.minus(a, b);
        assertEquals(-4, minus.get(0, 0), 1e-10); // 1 - 5
        assertEquals(-4, minus.get(0, 1), 1e-10); // 2 - 6
        
        // Test element-wise multiplication
        SimpleMatrix elemMult = ops.elementMult(a, b);
        assertEquals(5, elemMult.get(0, 0), 1e-10); // 1 * 5
        assertEquals(12, elemMult.get(0, 1), 1e-10); // 2 * 6
        
        // Test function application
        SimpleMatrix func = ops.applyFunction(a, x -> x * 2);
        assertEquals(2, func.get(0, 0), 1e-10); // 1 * 2
        assertEquals(4, func.get(0, 1), 1e-10); // 2 * 2
    }
}