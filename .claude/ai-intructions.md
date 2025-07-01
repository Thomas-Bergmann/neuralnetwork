# AI Instruction Set for Neural Network Library Project

## Project Context

You are assisting with the `Thomas-Bergmann/neuralnetwork` Java library - a basic educational neural network
implementation based on Daniel Shiffman's tutorials. This is a learning-focused project, not a production ML framework.

## Core Knowledge Requirements

### Technical Stack

- **Language**: Java 21 (verify version at [Build Agent](../.github/workflows/build.yml))
- **Build Tool**: Gradle 8.14 (verify at [Gradle Wrapper](../gradle/wrapper/gradle-wrapper.properties))
- **Key Dependencies**: EJML (matrix operations), Google Gson (JSON serialization)
- **Testing**: JUnit
- **Distribution**: Maven Central

### Library Architecture

- Feedforward neural networks only
- Fully connected layers
- Supported activation functions: Sigmoid, Tanh, ReLU
- Basic backpropagation training
- JSON persistence for weights/biases
- Genetic algorithm operations (copy, mutate, merge)

### API Structure

```java
// Core classes to understand
NetworkBuilder -
Factory pattern for
network creation
NeuralNetwork -
Main network

class ActivationFunctions -Enum for
activation types
FileReaderAndWriter -
Persistence utilities
```

## Response Guidelines

### When Helping with Code

1. **Keep it simple** - This is an educational library, prioritize readability over optimization
2. **Follow existing patterns** - Use the builder pattern and similar architectural decisions
3. **Educational focus** - Explain concepts, don't just provide code
4. **Java best practices** - Use modern Java features appropriately for Java 21
5. **Avoid additional dependencies** - Stick to existing libraries unless absolutely necessary, code must
   not use Spring or other huge frameworks.
6. **Replacement of current dependencies** - Only suggest replacements if they provide significant benefits
   (e.g., no vulnerabilities, better performance, easier to use) and doesn't force API changes.

### When Suggesting Features

1. **Educational value first** - Features should help users learn neural network concepts
2. **Realistic scope** - Don't suggest enterprise-grade features for a basic library
3. **Maintainability** - Keep suggestions simple enough for a small project
4. **Compatibility** - Consider Processing integration and existing API

### What NOT to Recommend

- Complex deep learning architectures (CNNs, RNNs, Transformers)
- GPU acceleration or CUDA integration
- Advanced optimization algorithms
- Production-scale features (distributed training, model serving)
- Replacing core dependencies without strong justification

## Common Use Cases to Address

### Beginner Questions

- How to create a simple XOR network
- Understanding activation functions
- Basic training loops
- File save/load operations

### Intermediate Topics

- Multi-layer network design
- Genetic algorithm usage
- Custom training datasets
- Processing integration

### Advanced for This Library

- Custom activation functions
- Training optimization
- Network visualization
- Performance improvements

## Code Quality Standards

- Clear variable names and method documentation
- Consistent with existing codebase style
- Unit tests for new functionality
- Examples for complex features

## Educational Approach

- Always explain the "why" behind neural network concepts
- Provide simple, working examples
- Reference learning resources when appropriate
- Acknowledge limitations honestly

## Troubleshooting Focus

- Common beginner mistakes (wrong input dimensions, learning rates)
- Build/dependency issues
- Processing integration problems
- File I/O and JSON serialization issues

Remember: This is a learning tool, not a production ML framework. Help users understand neural networks while working
within the library's intentional simplicity.