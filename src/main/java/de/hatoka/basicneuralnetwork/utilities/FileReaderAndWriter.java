package de.hatoka.basicneuralnetwork.utilities;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Path;

import org.ejml.simple.SimpleMatrix;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.stream.JsonReader;

import de.hatoka.basicneuralnetwork.NeuralNetwork;

/**
 * Created by KimFeichtinger on 26.04.18.
 * FileReaderAndWriter is responsible for writing and reading neural networks.
 * {@link NeuralNetwork} is annotated with gson to support serialization.
 */
public class FileReaderAndWriter
{
    /**
     * Writes a neural network to file
     * @param nn network
     * @param file file location
     * @throws IOException in case writing network to file fails
     */
    public void write(NeuralNetwork nn, Path file) throws IOException
    {
        try (FileWriter fw = new FileWriter(file.toFile()))
        {
            fw.write(asJson(nn));
            fw.flush();
        }
    }

    /**
     * Read neural network from file
     * @param file file location
     * @return neural network
     * @throws IOException in case reading from file fails
     */
    public NeuralNetwork read(Path file) throws IOException
    {
        try (JsonReader jsonReader = new JsonReader(new FileReader(file.toFile())))
        {
            NeuralNetwork nn = getGson().fromJson(jsonReader, NeuralNetwork.class);
            nn.afterLoad();
            return nn;
        }
    }

    /**
     * Read neural network from resource (application can provide trained networks
     * @param input input stream from resource
     * @return neural network
     * @throws IOException in case reading from input stream fails
     */
    public NeuralNetwork read(InputStream input) throws IOException
    {
        try (JsonReader jsonReader = new JsonReader(new InputStreamReader(input)))
        {
            NeuralNetwork nn = getGson().fromJson(jsonReader, NeuralNetwork.class);
            nn.afterLoad();
            return nn;
        }
    }
    
    public String asJson(NeuralNetwork nn)
    {
        return getGson().toJson(nn);
    }
    /**
     * @return Gson via GsonBuilder with all the needed adapters added
     */
    private Gson getGson()
    {
        return new GsonBuilder().registerTypeAdapter(SimpleMatrix.class, new SimpleMatrixAdapter())
                                .setPrettyPrinting()
                                .excludeFieldsWithoutExposeAnnotation().create();
    }

}
