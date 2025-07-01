package de.hatoka.basicneuralnetwork;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

import de.hatoka.basicneuralnetwork.activationfunctions.ActivationFunctions;

class NetworkConfigurationTest
{
    NetworkConfiguration config1 = new NetworkConfiguration(2, 3, new int[] {1,3,4}, 0.1, ActivationFunctions.SIGMOID, 123456L, false, 4);
    NetworkConfiguration config2 = new NetworkConfiguration(2, 3, new int[] {1,3,4}, 0.1, ActivationFunctions.SIGMOID, 123456L, false, 4);

    @Test
    void testHashCode()
    {
        assertEquals(config1.hashCode(), config2.hashCode());
    }

    @Test
    void testEquals()
    {
        assertEquals(config1, config2);
    }

}
