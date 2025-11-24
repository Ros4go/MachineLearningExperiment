using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Perceptron
{
    public List<InputPerceptron> inputs;
    public float state;
    public float error;

    public ThresholdType thresholdType;

    float gain = 0.01f;

    public Perceptron(ThresholdType _thresholdType, float _gain)
    {
        gain = _gain;
        thresholdType = _thresholdType;
    }

    public void Feedforward()
    {
        float sum = 0;
        for (int i = 0; i < inputs.Count; i++)
        {
            sum += inputs[i].inputPerceptron.state * inputs[i].weight;
        }

        this.state = Threshold(sum);
    }

    public void AdjustWeights(float currentError)
    {
        for (int i = 0; i < inputs.Count; i++)
        {
            state = inputs[i].inputPerceptron.state;
            float deltaWeight = gain * currentError * state;
            inputs[i].SetWeigth(inputs[i].weight + deltaWeight);
        }

        error = currentError;
    }

    public float GetIncomingWeight(Perceptron perceptron)
    {
        for (int i = 0; i < inputs.Count; i++)
        {
            if (inputs[i].inputPerceptron == perceptron)
            {
                return inputs[i].weight;
            }
        }

        return 0f;
    }

    public float Threshold(float input)
    {
        switch (thresholdType)
        {
            case ThresholdType.Tanh:
                return (Mathf.Exp(input) - Mathf.Exp(-input)) / (Mathf.Exp(input) + Mathf.Exp(-input));
            case ThresholdType.ReLU:
                return Mathf.Max(0, input);
            case ThresholdType.Sigmoid:
                return 1f / (1f + Mathf.Exp(-input));
        }

        return -1f;
    }
    
    public float ActivationDerivative(float state)
    {
        switch (thresholdType)
        {
            case ThresholdType.Tanh:
                return 1f - state * state;
            case ThresholdType.Sigmoid:
                return state * (1f - state);
            case ThresholdType.ReLU:
                return state > 0f ? 1f : 0f;
        }
        return -1f;
    }
}

public class  InputPerceptron {
    public Perceptron inputPerceptron;
    public float weight;
  
    public InputPerceptron(Perceptron inputPerceptron, float weight) {
        this.inputPerceptron = inputPerceptron;
        this.weight = weight;
    }

    public void SetWeigth (float newWeight) 
    {
        this.weight = newWeight;
    }
}