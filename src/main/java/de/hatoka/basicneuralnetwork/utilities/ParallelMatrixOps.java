package de.hatoka.basicneuralnetwork.utilities;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;

import org.ejml.simple.SimpleMatrix;

/**
 * Parallel matrix operations for neural network training acceleration.
 * Provides parallel alternatives to standard matrix operations when beneficial.
 */
public class ParallelMatrixOps implements MatrixOps {
    
    private static final int MIN_ELEMENTS_FOR_PARALLEL = 1000;
    private final ForkJoinPool threadPool;
    
    /**
     * Constructor with custom thread pool
     * @param threadPool the ForkJoinPool to use for parallel operations
     */
    public ParallelMatrixOps(ForkJoinPool threadPool) {
        this.threadPool = threadPool;
    }
    
    /**
     * Constructor that creates a new thread pool with specified parallelism
     * @param parallelism number of threads to use
     */
    public ParallelMatrixOps(int parallelism) {
        this.threadPool = new ForkJoinPool(parallelism);
    }
    
    @Override
    public SimpleMatrix mult(SimpleMatrix a, SimpleMatrix b) {
        if (shouldUseParallel(a, b)) {
            return threadPool.invoke(new MatrixMultTask(a, b, 0, a.numRows()));
        } else {
            return a.mult(b);
        }
    }
    
    @Override
    public SimpleMatrix plus(SimpleMatrix a, SimpleMatrix b) {
        if (shouldUseParallel(a)) {
            SimpleMatrix result = new SimpleMatrix(a.numRows(), a.numCols());
            threadPool.invoke(new ElementWiseTask(a, b, result, ElementWiseTask.Operation.ADD, 0, a.getNumElements()));
            return result;
        } else {
            return a.plus(b);
        }
    }
    
    @Override
    public SimpleMatrix minus(SimpleMatrix a, SimpleMatrix b) {
        if (shouldUseParallel(a)) {
            SimpleMatrix result = new SimpleMatrix(a.numRows(), a.numCols());
            threadPool.invoke(new ElementWiseTask(a, b, result, ElementWiseTask.Operation.SUBTRACT, 0, a.getNumElements()));
            return result;
        } else {
            return a.minus(b);
        }
    }
    
    @Override
    public SimpleMatrix elementMult(SimpleMatrix a, SimpleMatrix b) {
        if (shouldUseParallel(a)) {
            SimpleMatrix result = new SimpleMatrix(a.numRows(), a.numCols());
            threadPool.invoke(new ElementWiseTask(a, b, result, ElementWiseTask.Operation.ELEMENT_MULT, 0, a.getNumElements()));
            return result;
        } else {
            return a.elementMult(b);
        }
    }
    
    @Override
    public SimpleMatrix applyFunction(SimpleMatrix input, java.util.function.DoubleUnaryOperator function) {
        if (input.getNumElements() > MIN_ELEMENTS_FOR_PARALLEL) {
            return threadPool.invoke(new FunctionApplicationTask(input, function, 0, input.getNumElements()));
        } else {
            SimpleMatrix result = new SimpleMatrix(input.numRows(), input.numCols());
            for (int i = 0; i < input.getNumElements(); i++) {
                result.set(i, function.applyAsDouble(input.get(i)));
            }
            return result;
        }
    }
    
    @Override
    public void close() {
        if (threadPool != null && !threadPool.isShutdown()) {
            threadPool.shutdown();
        }
    }
    
    private static boolean shouldUseParallel(SimpleMatrix a, SimpleMatrix b) {
        return a.getNumElements() * b.getNumElements() > MIN_ELEMENTS_FOR_PARALLEL * MIN_ELEMENTS_FOR_PARALLEL;
    }
    
    private static boolean shouldUseParallel(SimpleMatrix a) {
        return a.getNumElements() > MIN_ELEMENTS_FOR_PARALLEL;
    }
    
    /**
     * Recursive task for parallel matrix multiplication
     */
    private static class MatrixMultTask extends RecursiveTask<SimpleMatrix> {
        private final SimpleMatrix a, b;
        private final int startRow, endRow;
        private static final int THRESHOLD = 64;
        
        MatrixMultTask(SimpleMatrix a, SimpleMatrix b, int startRow, int endRow) {
            this.a = a;
            this.b = b;
            this.startRow = startRow;
            this.endRow = endRow;
        }
        
        @Override
        protected SimpleMatrix compute() {
            if (endRow - startRow <= THRESHOLD) {
                // Compute directly for small ranges
                SimpleMatrix result = new SimpleMatrix(endRow - startRow, b.numCols());
                for (int i = startRow; i < endRow; i++) {
                    for (int j = 0; j < b.numCols(); j++) {
                        double sum = 0.0;
                        for (int k = 0; k < a.numCols(); k++) {
                            sum += a.get(i, k) * b.get(k, j);
                        }
                        result.set(i - startRow, j, sum);
                    }
                }
                return result;
            } else {
                // Split the work
                int mid = (startRow + endRow) / 2;
                MatrixMultTask task1 = new MatrixMultTask(a, b, startRow, mid);
                MatrixMultTask task2 = new MatrixMultTask(a, b, mid, endRow);
                
                task1.fork();
                SimpleMatrix result2 = task2.compute();
                SimpleMatrix result1 = task1.join();
                
                // Combine results vertically
                return result1.combine(result1.numRows(), 0, result2);
            }
        }
    }
    
    /**
     * Recursive task for parallel element-wise operations
     */
    private static class ElementWiseTask extends RecursiveTask<Void> {
        enum Operation { ADD, SUBTRACT, ELEMENT_MULT }
        
        private final SimpleMatrix a, b, result;
        private final Operation op;
        private final int start, end;
        private static final int THRESHOLD = 1000;
        
        ElementWiseTask(SimpleMatrix a, SimpleMatrix b, SimpleMatrix result, Operation op, int start, int end) {
            this.a = a;
            this.b = b;
            this.result = result;
            this.op = op;
            this.start = start;
            this.end = end;
        }
        
        @Override
        protected Void compute() {
            if (end - start <= THRESHOLD) {
                // Direct computation for small ranges
                for (int i = start; i < end; i++) {
                    double valueA = a.get(i);
                    double valueB = b.get(i);
                    double resultValue = switch (op) {
                        case ADD -> valueA + valueB;
                        case SUBTRACT -> valueA - valueB;
                        case ELEMENT_MULT -> valueA * valueB;
                    };
                    result.set(i, resultValue);
                }
                return null;
            } else {
                // Split work and compute in parallel
                int mid = (start + end) / 2;
                ElementWiseTask task1 = new ElementWiseTask(a, b, result, op, start, mid);
                ElementWiseTask task2 = new ElementWiseTask(a, b, result, op, mid, end);
                
                task1.fork();
                task2.compute();
                task1.join();
                return null;
            }
        }
    }
    
    /**
     * Recursive task for parallel function application
     */
    private static class FunctionApplicationTask extends RecursiveTask<SimpleMatrix> {
        private final SimpleMatrix input;
        private final java.util.function.DoubleUnaryOperator function;
        private final int start, end;
        private static final int THRESHOLD = 1000;
        
        FunctionApplicationTask(SimpleMatrix input, java.util.function.DoubleUnaryOperator function, int start, int end) {
            this.input = input;
            this.function = function;
            this.start = start;
            this.end = end;
        }
        
        @Override
        protected SimpleMatrix compute() {
            if (end - start <= THRESHOLD) {
                SimpleMatrix result = new SimpleMatrix(input.numRows(), input.numCols());
                for (int i = start; i < end; i++) {
                    result.set(i, function.applyAsDouble(input.get(i)));
                }
                return result;
            } else {
                int mid = (start + end) / 2;
                FunctionApplicationTask task1 = new FunctionApplicationTask(input, function, start, mid);
                FunctionApplicationTask task2 = new FunctionApplicationTask(input, function, mid, end);
                
                task1.fork();
                SimpleMatrix result2 = task2.compute();
                SimpleMatrix result1 = task1.join();
                
                // Combine results
                SimpleMatrix combined = new SimpleMatrix(input.numRows(), input.numCols());
                for (int i = 0; i < result1.getNumElements(); i++) {
                    combined.set(i, result1.get(i));
                }
                for (int i = 0; i < result2.getNumElements(); i++) {
                    combined.set(mid - start + i, result2.get(i));
                }
                return combined;
            }
        }
    }
}