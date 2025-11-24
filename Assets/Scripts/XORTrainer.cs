using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Collections.Generic;
using System;
using System.Globalization;

public class XORTrainer : MonoBehaviour
{
    private float[,] trainingSet;
    private float[] refOutputs;

    private MLPNetwork MLPNet;

    private Slider inputLabel_1;
    private Slider inputLabel_2;
    private Slider inputLabel_3;

    private TMP_Text outputLabel;
    private TMP_InputField iterationsInput;
    private TMP_Text operatorText;

    public enum Set {XOR, AND, RESULT};
    [SerializeField] Set trainingSetName = Set.XOR;
    [SerializeField] int[] perceptronNumber = new int[] { 5, 3 };
    [SerializeField] public bool useGeneticAlgorithm = false;
    [SerializeField] public ThresholdType thresholdTypeGenetic = ThresholdType.Sigmoid;
    [SerializeField] public int defaultNbIteration = 100;

    private GeneticTrainer geneticTrainer;
    private bool geneticInitialized = false;

    void Start ()
    {
        inputLabel_1 = GameObject.Find("Life").GetComponent<Slider>();
        inputLabel_2 = GameObject.Find("Ammo").GetComponent<Slider>();
        inputLabel_3 = GameObject.Find("Enemy").GetComponent<Slider>();
        outputLabel = GameObject.Find("Output/Text (TMP)").GetComponent<TMP_Text>();
        iterationsInput = GameObject.Find("Iterations").GetComponent<TMP_InputField>();
        operatorText = GameObject.Find("Operator Button/Text (TMP)").GetComponent<TMP_Text>();

        if (trainingSetName == Set.XOR)
        {
            trainingSet = new float[4,2];
            refOutputs = new float[4];

            trainingSet[0, 0] = 0f; trainingSet[0, 1] = 0f;
            trainingSet[1, 0] = 0f; trainingSet[1, 1] = 1f;
            trainingSet[2, 0] = 1f; trainingSet[2, 1] = 0f;
            trainingSet[3, 0] = 1f; trainingSet[3, 1] = 1f;

            refOutputs[0] = 0f;
            refOutputs[1] = 1f;
            refOutputs[2] = 1f;
            refOutputs[3] = 0f;

            operatorText.text = "XOR";

            if (!useGeneticAlgorithm)
            {
                MLPNet = GetComponent<MLPNetwork>();
                MLPNet.Init(2, perceptronNumber, 1);
            }
        }
        else if (trainingSetName == Set.AND)
        {
            trainingSet = new float[4,2];
            refOutputs = new float[4];

            trainingSet[0, 0] = 0f; trainingSet[0, 1] = 0f;
            trainingSet[1, 0] = 0f; trainingSet[1, 1] = 1f;
            trainingSet[2, 0] = 1f; trainingSet[2, 1] = 0f;
            trainingSet[3, 0] = 1f; trainingSet[3, 1] = 1f;

            refOutputs[0] = 0f;
            refOutputs[1] = 0f;
            refOutputs[2] = 0f;
            refOutputs[3] = 1f;

            operatorText.text = "AND";

            if (!useGeneticAlgorithm)
            {
                MLPNet = GetComponent<MLPNetwork>();
                MLPNet.Init(2, perceptronNumber, 1);
            }
        }
        else if (trainingSetName == Set.RESULT)
        {
            trainingSet = new float[7, 3];
            refOutputs = new float[7];

            trainingSet[0, 0] = 0f; trainingSet[0, 1] = 0f; trainingSet[0, 2] = 0f; refOutputs[0] = 0f;
            trainingSet[1, 0] = 0f; trainingSet[1, 1] = 0f; trainingSet[1, 2] = 1f; refOutputs[1] = 0f;
            trainingSet[2, 0] = 0f; trainingSet[2, 1] = 1f; trainingSet[2, 2] = 0f; refOutputs[2] = 0.5f;
            trainingSet[3, 0] = 0f; trainingSet[3, 1] = 1f; trainingSet[3, 2] = 1f; refOutputs[3] = 0f;
            trainingSet[4, 0] = 1f; trainingSet[4, 1] = 0f; trainingSet[4, 2] = 0f; refOutputs[4] = 0.5f;
            trainingSet[5, 0] = 1f; trainingSet[5, 1] = 1f; trainingSet[5, 2] = 0f; refOutputs[5] = 1f;
            trainingSet[6, 0] = 1f; trainingSet[6, 1] = 1f; trainingSet[6, 2] = 1f; refOutputs[6] = 1f;

            operatorText.text = "RESULT";

            if (!useGeneticAlgorithm)
            {
                MLPNet = GetComponent<MLPNetwork>();
                MLPNet.Init(3, perceptronNumber, 1);
            }
        }
    }

    public void ComputeOutput()
    {
        List<float> inputList = new List<float>();

        if (useGeneticAlgorithm && geneticTrainer != null)
        {
            foreach (var net in geneticTrainer.GetPopulation())
            {
                inputList.Clear();
                if (net.inputPerceptrons.Count >= 1)
                    inputList.Add(inputLabel_1.value);
                if (net.inputPerceptrons.Count >= 2)
                    inputList.Add(inputLabel_2.value);
                if (net.inputPerceptrons.Count >= 3)
                    inputList.Add(inputLabel_3.value);

                net.GenerateOutput(inputList);
            }

            outputLabel.text = geneticTrainer.GetBestNetwork().lastOutput.ToString();

            var visualizer = GetComponent<NetworkVisualizer>();
            if (visualizer != null)
            {
                visualizer.ShowPopulationOutputs(geneticTrainer.GetPopulation());
            }
        }
        else if (MLPNet != null)
        {
            if (MLPNet.inputPerceptrons.Count >= 1)
                inputList.Add(inputLabel_1.value);
            if (MLPNet.inputPerceptrons.Count >= 2)
                inputList.Add(inputLabel_2.value);
            if (MLPNet.inputPerceptrons.Count >= 3)
                inputList.Add(inputLabel_3.value);

            MLPNet.GenerateOutput(inputList);
            outputLabel.text = MLPNet.GetOutputs()[0].ToString("F3");
        }
    }

    public void TrainNetwork()
    {
        if (useGeneticAlgorithm)
        {
            TrainWithGeneticAlgorithm();
        }
        else
        {
            TrainWithBackpropagation();
        }
    }

    private void TrainWithGeneticAlgorithm()
    {
        if (!geneticInitialized)
        {
            geneticTrainer = new GeneticTrainer(
                thresholdType: thresholdTypeGenetic,
                populationSize: 50,
                inputCount: trainingSet.GetLength(1),
                hiddenLayerSizes: perceptronNumber,
                outputCount: 1,
                fitnessFunction: EvaluateNetwork
            );
            geneticInitialized = true;
        }

        int nbIterations;
        if (!int.TryParse(iterationsInput.text, out nbIterations))
            nbIterations = defaultNbIteration;

        for (int i = 0; i < nbIterations; i++)
            geneticTrainer.Evolve();

        MLPNet = geneticTrainer.GetBestNetwork();
        GetComponent<NetworkVisualizer>()?.UpdateConnectionColors();
    }

    private float EvaluateNetwork(MLPNetwork net)
    {
        int evalIterations = 1;
        float totalError = 0f;
        List<float> inputList = new List<float>();
        List<float> outputList = new List<float>();

        for (int iter = 0; iter < evalIterations; iter++)
        {
            for (int i = 0; i < refOutputs.Length; i++)
            {
                inputList.Clear();
                for (int j = 0; j < trainingSet.GetLength(1); j++)
                    inputList.Add(trainingSet[i, j]);

                outputList.Clear();
                outputList.Add(refOutputs[i]);

                net.GenerateOutput(inputList);
                float output = net.GetOutputs()[0];
                float error = refOutputs[i] - output;
                totalError += error * error;
            }
        }

        return -totalError / evalIterations;
    }

    public void TrainWithBackpropagation()
    {
        int nbIterations, iterCounter = 0;

        if (int.TryParse(iterationsInput.text, out nbIterations) == false)
        {
            Debug.LogWarning("invalid NbIterations - setting 1000 iterations by default");
            nbIterations = defaultNbIteration;
        }
        List<float> inputList = new List<float>();
        List<float> outputList = new List<float>();

        while (iterCounter++ != nbIterations)
        {
            for (int i = 0; i < refOutputs.Length; i++)
            {
                inputList.Clear();
                for (int j = 0; j < trainingSet.GetLength(1); j++)
                {
                    inputList.Add(trainingSet[i, j]);
                }

                outputList.Clear();
                outputList.Add(refOutputs[i]);
                MLPNet.LearnPattern(inputList, outputList);
            }
        }

        GetComponent<NetworkVisualizer>()?.UpdateConnectionColors();
        Debug.Log("Network trained " + nbIterations + " times!");
    }
}
