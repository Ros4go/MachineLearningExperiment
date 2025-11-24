using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum ThresholdType {Sigmoid, Tanh, ReLU};

public class MLPNetwork : MonoBehaviour
{
    [SerializeField] public ThresholdType thresholdInputOutputType = ThresholdType.Sigmoid;
    [SerializeField] public ThresholdType thresholdType = ThresholdType.Sigmoid;
    [SerializeField] float gain = 0.01f;

    List<float> MLPNOutputList = new List<float>();

    public List<Perceptron> inputPerceptrons;
    public List<List<Perceptron>> hiddenLayers;
    public List<Perceptron> outputPerceptrons;

    public float lastOutput;

    public void Init(int inputCount, int[] hiddenLayerSizes, int outputCount = 1)
    {
        inputPerceptrons = new List<Perceptron>();
        hiddenLayers = new List<List<Perceptron>>();
        outputPerceptrons = new List<Perceptron>();

        for (int i = 0; i < inputCount; i++)
        {
            inputPerceptrons.Add(new Perceptron(thresholdInputOutputType, gain));
        }

        List<Perceptron> previousLayer = inputPerceptrons;
        foreach (int layerSize in hiddenLayerSizes)
        {
            List<Perceptron> currentLayer = new List<Perceptron>();
            for (int i = 0; i < layerSize; i++)
            {
                Perceptron p = new Perceptron(thresholdType, gain);
                p.inputs = new List<InputPerceptron>();
                foreach (var prev in previousLayer)
                {
                    p.inputs.Add(new InputPerceptron(prev, GetInitialWeight(inputCount, thresholdType)));
                }
                currentLayer.Add(p);
            }
            hiddenLayers.Add(currentLayer);
            previousLayer = currentLayer;
        }

        for (int i = 0; i < outputCount; i++)
        {
            Perceptron p = new Perceptron(thresholdInputOutputType, gain);
            p.inputs = new List<InputPerceptron>();
            foreach (var prev in previousLayer)
            {
                p.inputs.Add(new InputPerceptron(prev, GetInitialWeight(inputCount, thresholdInputOutputType)));
            }
            outputPerceptrons.Add(p);
        }
    }

    public static float GetInitialWeight(int n, ThresholdType thresholdType)
    {
        float limit;

        switch (thresholdType)
        {
            case ThresholdType.Sigmoid:
                limit = 0.3f;
                break;
            case ThresholdType.Tanh:
                limit = Mathf.Sqrt(1f / n);
                break;
            case ThresholdType.ReLU:
                limit = Mathf.Sqrt(2f / n);
                break;
            default:
                limit = 0.1f;
                break;
        }

        return UnityEngine.Random.Range(-limit, limit);
    }

    public List<float> GetWeights()
    {
        List<float> weights = new List<float>();

        foreach (var layer in hiddenLayers)
        {
            foreach (var perceptron in layer)
            {
                foreach (var input in perceptron.inputs)
                {
                    weights.Add(input.weight);
                }
            }
        }

        foreach (var perceptron in outputPerceptrons)
        {
            foreach (var input in perceptron.inputs)
            {
                weights.Add(input.weight);
            }
        }

        return weights;
    }

    public void SetWeights(List<float> weights)
    {
        int index = 0;

        foreach (var layer in hiddenLayers)
        {
            foreach (var perceptron in layer)
            {
                for (int i = 0; i < perceptron.inputs.Count; i++)
                {
                    perceptron.inputs[i].SetWeigth(weights[index++]);
                }
            }
        }

        foreach (var perceptron in outputPerceptrons)
        {
            for (int i = 0; i < perceptron.inputs.Count; i++)
            {
                perceptron.inputs[i].SetWeigth(weights[index++]);
            }
        }
    }

    public void CopyWeightsFrom(MLPNetwork source)
    {
        var sourceLayers = source.hiddenLayers;
        var targetLayers = this.hiddenLayers;

        for (int l = 0; l < targetLayers.Count; l++)
        {
            for (int p = 0; p < targetLayers[l].Count; p++)
            {
                var targetPerceptron = targetLayers[l][p];
                var sourcePerceptron = sourceLayers[l][p];

                for (int i = 0; i < targetPerceptron.inputs.Count; i++)
                {
                    targetPerceptron.inputs[i].SetWeigth(sourcePerceptron.inputs[i].weight);
                }
            }
        }

        for (int p = 0; p < outputPerceptrons.Count; p++)
        {
            var targetP = outputPerceptrons[p];
            var sourceP = source.outputPerceptrons[p];

            for (int i = 0; i < targetP.inputs.Count; i++)
            {
                targetP.inputs[i].SetWeigth(sourceP.inputs[i].weight);
            }
        }
    }

    public void GenerateOutput(List<float> inputList)
    {
        for (int i = 0; i < inputPerceptrons.Count; i++)
        {
            inputPerceptrons[i].state = inputList[i];
        }

        foreach (var layer in hiddenLayers)
        {
            foreach (var perceptron in layer)
            {
                perceptron.Feedforward();
            }
        }

        foreach (var perceptron in outputPerceptrons)
        {
            perceptron.Feedforward();
        }

        lastOutput = outputPerceptrons[0].state;
    }

    void Backprop(List<float> outputList)
    {
        for (int i = 0; i < outputPerceptrons.Count; i++)
        {
            Perceptron perceptron = outputPerceptrons[i];
            float state = perceptron.state;
            float error = perceptron.ActivationDerivative(state) * (outputList[i] - state);
            perceptron.AdjustWeights(error);
        }

        for (int l = hiddenLayers.Count - 1; l >= 0; l--)
        {
            var currentLayer = hiddenLayers[l];
            foreach (var perceptron in currentLayer)
            {
                float state = perceptron.state;
                float sum = 0;
                List<Perceptron> nextLayer = (l == hiddenLayers.Count - 1) ? outputPerceptrons : hiddenLayers[l + 1];

                foreach (var next in nextLayer)
                {
                    float weight = next.GetIncomingWeight(perceptron);
                    sum += weight * next.error;
                }   

                float error = perceptron.ActivationDerivative(state) * sum;
                perceptron.AdjustWeights(error);
            }
        }
    }

    public List<float> GetOutputs()
    {
        MLPNOutputList.Clear();

        for (int i = 0; i < outputPerceptrons.Count; i++)
        {
            MLPNOutputList.Add(outputPerceptrons[i].state);
        }
        return MLPNOutputList;
    }

    public void LearnPattern(List<float> inputList, List<float> outputList)
    {
        GenerateOutput(inputList);
        Backprop(outputList);
    }
}
