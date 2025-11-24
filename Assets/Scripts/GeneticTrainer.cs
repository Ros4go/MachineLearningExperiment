using System;
using System.Collections.Generic;
using UnityEngine;

public class GeneticTrainer
{
    public int populationSize;
    public int eliteCount;
    public float mutationRate;
    public float crossoverRate;
    public Func<MLPNetwork, float> fitnessFunction;

    private int[] hiddenLayerSizes;
    private int inputCount, outputCount;

    private List<MLPNetwork> population = new List<MLPNetwork>();
    private MLPNetwork bestNetwork;

    public ThresholdType thresholdType = ThresholdType.Sigmoid;

    public GeneticTrainer(ThresholdType thresholdType, int populationSize, int inputCount, int[] hiddenLayerSizes, int outputCount,
                          int eliteCount = 20, float mutationRate = 0.4f, float crossoverRate = 0.5f,
                          Func<MLPNetwork, float> fitnessFunction = null)
    {
        this.thresholdType = thresholdType;
        this.populationSize = populationSize;
        this.eliteCount = eliteCount;
        this.mutationRate = mutationRate;
        this.crossoverRate = crossoverRate;
        this.inputCount = inputCount;
        this.outputCount = outputCount;
        this.hiddenLayerSizes = hiddenLayerSizes;
        this.fitnessFunction = fitnessFunction;

        InitPopulation();
    }

    private void InitPopulation()
    {
        population.Clear();
        for (int i = 0; i < populationSize; i++)
        {
            var net = new GameObject("MLP").AddComponent<MLPNetwork>();
            net.thresholdInputOutputType = thresholdType;
            net.thresholdType = thresholdType;
            net.Init(inputCount, hiddenLayerSizes, outputCount);
            population.Add(net);
        }
    }

    public void Evolve()
    {
        if (fitnessFunction == null)
        {
            Debug.LogError("Fitness function not set.");
            return;
        }

        population.Sort((a, b) => fitnessFunction(b).CompareTo(fitnessFunction(a)));
        bestNetwork = population[0];
        List<MLPNetwork> newPopulation = new List<MLPNetwork>();

        for (int i = 0; i < eliteCount; i++)
        {
            newPopulation.Add(CloneNetwork(population[i]));
        }

        while (newPopulation.Count < populationSize - 1)
        {
            var parent = population[UnityEngine.Random.Range(0, eliteCount)];
            var child = CloneNetwork(parent);
            Mutate(child, population.GetRange(0, eliteCount));
            newPopulation.Add(child);
        }

        var newRandom = new GameObject("MLP_Random").AddComponent<MLPNetwork>();
        newRandom.thresholdInputOutputType = thresholdType;
        newRandom.thresholdType = thresholdType;
        newRandom.Init(inputCount, hiddenLayerSizes, outputCount);
        newPopulation.Add(newRandom);

        foreach (var net in population)
            UnityEngine.Object.Destroy(net.gameObject);

        population = newPopulation;
        population.Sort((a, b) => fitnessFunction(b).CompareTo(fitnessFunction(a)));
        bestNetwork = population[0];
    }

    private MLPNetwork CloneNetwork(MLPNetwork original)
    {
        var clone = new GameObject("MLP_Clone").AddComponent<MLPNetwork>();
        clone.thresholdInputOutputType = thresholdType;
        clone.thresholdType = thresholdType;
        clone.Init(inputCount, hiddenLayerSizes, outputCount);
        clone.CopyWeightsFrom(original);
        return clone;
    }

    private void Mutate(MLPNetwork network, List<MLPNetwork> elites)
    {
        List<float> weights = network.GetWeights();

        for (int i = 0; i < weights.Count; i++)
        {
            float rand = UnityEngine.Random.value;
            if (rand < mutationRate)
            {
                weights[i] = UnityEngine.Random.Range(-1f, 1f);
            }
            else if (rand < mutationRate + crossoverRate)
            {
                var other = elites[UnityEngine.Random.Range(0, elites.Count)];
                weights[i] = other.GetWeights()[i];
            }
        }

        network.SetWeights(weights);
    }

    public MLPNetwork GetBestNetwork()
    {
        return bestNetwork;
    }

    public List<MLPNetwork> GetPopulation()
    {
        return population;
    }
}
