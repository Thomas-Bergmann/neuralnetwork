# Basic Neural Network Library

* ![Build & Test](https://github.com/Thomas-Bergmann/neuralnetwork/actions/workflows/build.yml/badge.svg)
* ![Dependencies](https://github.com/Thomas-Bergmann/neuralnetwork/actions/workflows/depsubmission.yml/badge.svg)
* ![Publish](https://github.com/Thomas-Bergmann/neuralnetwork/actions/workflows/release.yml/badge.svg)

This is a very basic Java Neural Network library based on the one built by Daniel Shiffman in [this](https://www.youtube.com/watch?v=XJ7HLz9VYz0&list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh) playlist using the [Efficient Java Matrix Library](https://www.ejml.org) (EJML).

The library can also be used with [Processing](https://processing.org). Just download the jar-file (see below) and drag it into your sketch.

If you want to learn more about Neural Networks check out these YouTube-playlists:
- [Neural Networks - The Nature of Code](https://www.youtube.com/watch?v=XJ7HLz9VYz0&list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh) by The Coding Train (Daniel Shiffman)
- [Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) by 3Blue1Brown

The source is a fork from [Kim Marcel / Basic Neural Network](https://github.com/kim-marcel/basic_neural_network) with some smaller adjustments:
- library is available via [Maven Central]
- build is based on gradle and executed via github actions

## Features

- Neural Network with variable amounts of inputs, hidden nodes and outputs
- Multiple hidden layers
- Activation functions: Sigmoid, Tanh, ReLu
- Adjustable learning rate
- Fully connected
- Support for genetic algorithms: copy-, mutate-, and merge-functionality
- Save the weights and biases of a NN to a JSON-file
- Generate a NeuralNetwork-object from a JSON-file

## Getting Started

This section describes how a working copy of this project can be set up on your local machine for testing and development purposes. If you just want to use the library you can skip this part.

### Prerequisites
- [Java JDK 21](https://adoptium.net/temurin/releases/) needs to be installed and JAVA_HOME must be set

### Build and Testing
```
./gradlew test
```
Runs all JUnit-Tests specified in this project.

## Use the library
Dependencies
```gradle
implementation 'de.hatoka.neuralnetwork:neuralnetwork:2.0.0'
```

Constructors:
```java
import de.hatoka.basicneuralnetwork.NeuralNetwork;
import de.hatoka.basicneuralnetwork.NetworkBuilder;

// create simplest network with 2 inputs and 1 output
NeuralNetwork nn1 = NetworkBuilder.create(2,1).build();
// create network with 2 inputs, 2 hidden layer with 4 nodes and 1 output
NeuralNetwork nn0 = NetworkBuilder.create(2,1).setHiddenLayers(2, 4).build();
```

Train and guess:
```java
// Train the neural network with a training dataset (inputs and expected outputs)
nn.train(trainingDataInputs, trainingDataTargets);

// Guess for the given testing data is returned as an array (double[])
nn.guess(testingData);
```

Read and write from/to file:
```java
import de.hatoka.basicneuralnetwork.utilities.FileReaderAndWriter;
FileReaderAndWriter networkReaderWriter = new FileReaderAndWriter()
Path file = Files.createTempFile("neuro1_", ".json");
networkReaderWriter.write(network, file);

// Reads from a JSON-resource the nn-Data and returns a NeuralNetwork-object
NeuralNetwork networkViaResource = networkReaderWriter.read(this.getClass().getClassLoader().getResourceAsStream(resource));
// Load from a specifiy file
NeuralNetwork networkViaFile = networkReaderWriter.read(file);
```

Adjust the learning rate:
```java
NeuralNetwork nn = NetworkBuilder.create(2,1).setLearningRate(0.2).build();
```

Use different activation functions:
```java
// Set the activation function (By default Sigmoid will be used)
NeuralNetwork nn = NetworkBuilder.create(2,1).setActivationFunction(ActivationFunctions.TANH).build();
```

Use this library with genetic algorithms:
```java
// Make an exact and "independent" copy of a Neural Network
NeuralNetwork nn2 = nn1.copy();

// Merge the weights and biases of two Neural Networks with a ratio of 50:50
NeuralNetwork merged = nnA.merge(nnB);

// Merge the weights and biases of two Neural Networks with a custom ratio (here: 20:80)
NeuralNetwork merged = nnA.merge(nnB, 0.2);

// Mutate the weights and biases of a Neural Network with custom probability
nn.mutate(0.1);
```
## Examples

- [XOR solved with Basic Neural Network Library](https://github.com/kim-marcel/xor_with_nn)
- [Doodle Classification in Processing](https://github.com/kim-marcel/doodle_classifier)

## TODO

- implement softmax
- add more functionality for genetic algorithms (e.g. different merge functions,...)
- Implement more Unit-Tests (increase test coverage)
- Javadoc documentation
- weights and biases should get normalized
- more examples

If you have any other suggestions on what should be done, feel free to open an issue or add it to this list.

If you want to contribute by implementing any of these features or your own ideas please do so and send me a pull request.

## Libraries & Tools used in this project

- [EJML](https://www.ejml.org) used for Matrix math
- [Google Gson](https://github.com/google/gson) library
- [Gradle 8.1.1](https://docs.gradle.org/8.1.1/release-notes.html) build tooling
