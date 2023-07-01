package de.hatoka.basicneuralnetwork.utilities;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import de.hatoka.basicneuralnetwork.NetworkBuilder;
import de.hatoka.basicneuralnetwork.NeuralNetwork;

class FileReaderAndWriterTest
{
    private final FileReaderAndWriter underTest = new FileReaderAndWriter();
    private final List<Path> createdFiles = new ArrayList<>();
    private NeuralNetwork nn;

    @BeforeEach
    public void createPredefinedNetwork()
    {
        nn = NetworkBuilder.create(4, 3).setHiddenLayers(1, 3).setSeed(123456L).build();
        nn.train(new double[] { 0.1, 0.2, 0.3, 0.4 }, new double[] { 0.1, 0.2, 0.3 });
    }
    @AfterEach
    public void removeCreateFiles()
    {
        createdFiles.forEach(p -> p.toFile().delete());
        createdFiles.clear();
    }

    @Test
    void writeAndReadFileTest() throws IOException
    {
        Path file = Files.createTempFile("neuro1_", ".json");
        underTest.write(nn, file);
        createdFiles.add(file);
        assertEquals(nn, underTest.read(file));
    }

    @Test
    void readFromResourceTest() throws IOException
    {
        InputStream input = FileReaderAndWriter.class.getResourceAsStream("neuro1.json");
        NeuralNetwork stored = underTest.read(input);
        assertEquals(nn, stored);
    }
}
