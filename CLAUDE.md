# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Build and Test
- `./gradlew test` - Run all JUnit tests
- `./gradlew build` - Full build including tests, JAR creation, and code coverage
- `./gradlew jacocoTestReport` - Generate code coverage reports (outputs to `build/jacocoHtml/`)

### Running Single Tests
- `./gradlew test --tests "de.hatoka.basicneuralnetwork.NeuralNetworkTest"`
- `./gradlew test --tests "*MatrixUtilitiesTest"`

### Publishing (requires proper credentials)
- `./gradlew publish` - Publish to Maven Central (requires sonatype credentials)

## Architecture Overview

This is an educational Java neural network library with a simple, clean architecture focused on learning rather than production use.

### Core Components

**NetworkBuilder** (`src/main/java/de/hatoka/basicneuralnetwork/NetworkBuilder.java`)
- Factory pattern for creating neural networks with fluent API
- Configures inputs, outputs, hidden layers, activation functions, learning rate, and random seed
- Entry point: `NetworkBuilder.create(inputs, outputs).setHiddenLayers(layers, nodes).build()`

**NeuralNetwork** (`src/main/java/de/hatoka/basicneuralnetwork/NeuralNetwork.java`)
- Main network implementation with feedforward and backpropagation
- Key methods: `guess()` for inference, `train()` for single-sample training
- Genetic algorithm support: `copy()`, `merge()`, `mutate()`
- Parallel/sequential training via strategy pattern (MatrixOps)
- Resource cleanup via `close()` method for parallel networks

**NetworkConfiguration** (`src/main/java/de/hatoka/basicneuralnetwork/NetworkConfiguration.java`)
- Immutable value object storing network parameters
- Includes parallel training configuration (enabled/disabled, thread count)
- Factory methods for creating MatrixOps strategies
- Fully serializable with Gson for persistence

### Key Dependencies and Utilities

**EJML Integration** (`utilities/SimpleMatrixAdapter.java`, `utilities/MatrixUtilities.java`)
- All matrix operations use EJML's SimpleMatrix
- Custom Gson serialization adapter for matrix persistence
- Utility methods for common matrix operations

**Matrix Operations Strategy** (`utilities/MatrixOps.java`, `utilities/SequentialMatrixOps.java`, `utilities/ParallelMatrixOps.java`)
- Strategy pattern for matrix operations: `MatrixOps` interface with sequential and parallel implementations
- `SequentialMatrixOps`: Direct EJML operations (singleton, zero overhead)
- `ParallelMatrixOps`: Fork-join based parallel operations with automatic fallback for small matrices
- Automatic resource management via AutoCloseable

**Activation Functions** (`activationfunctions/`)
- Enum-based system: `ActivationFunctions.SIGMOID`, `TANH`, `RELU`
- Each function implements both forward and derivative calculations
- Extensible design for adding new activation functions

**File I/O** (`utilities/FileReaderAndWriter.java`)
- JSON serialization/deserialization of complete networks
- Supports both file-based and resource-based loading
- Uses Gson with custom adapters for SimpleMatrix

### Technical Constraints

- **Java 21** with modern language features (pattern matching in equals methods)
- **Educational focus**: Prioritize readability and learning over performance
- **Fully connected networks only**: No CNN, RNN, or other architectures
- **Single-threaded**: No parallel processing or GPU support
- **Fixed architecture**: Cannot modify network structure after creation

### Development Patterns

- **Builder pattern**: Preferred for object creation with many parameters
- **Immutable configurations**: NetworkConfiguration never changes after creation
- **Defensive copying**: Matrix operations create new instances rather than modifying in-place
- **Gson annotations**: Use `@Expose` for serialization control
- **Random seed control**: All randomization is deterministic for testing purposes

### Common Usage Patterns

**Basic Network Creation:**
```java
// Simple sequential network
NeuralNetwork basic = NetworkBuilder.create(2, 1)
    .setHiddenLayers(1, 4)
    .setLearningRate(0.1)
    .build();

// Large parallel network
NeuralNetwork parallel = NetworkBuilder.create(100, 50)
    .setHiddenLayers(3, 200)
    .enableParallelTraining(4)
    .build();
// Remember: parallel.close() when done
```

**Resource Management:**
- Sequential networks: No cleanup required
- Parallel networks: Always call `close()` to release thread pool resources
- Use try-with-resources for automatic cleanup: `try (NeuralNetwork nn = builder.enableParallelTraining().build())`

### Testing Strategy

- JUnit 5 with parameterized tests
- Test resources in `src/test/resources/` (e.g., sample network JSON files)
- Code coverage reporting with JaCoCo
- Focus on edge cases like dimension mismatches and boundary conditions
- Parallel training tests verify sequential/parallel equivalence

This library is designed for learning neural network fundamentals, not production ML workloads.