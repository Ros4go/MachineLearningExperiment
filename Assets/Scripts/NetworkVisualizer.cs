using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class NetworkVisualizer : MonoBehaviour
{
    public MLPNetwork network;
    public float verticalAreaSize = 10f;
    public float horizontalLayerSpacing = 3f;
    public float lineThickness = 0.05f;
    public GameObject neuronPrefab;

    private List<GameObject> neurons = new List<GameObject>();
    private List<LineRenderer> connections = new List<LineRenderer>();

    public float verticalNeuronSpacing = 1f;
    public Vector3 basePosition = Vector3.zero;

    public TMP_Text populationOutputLabel;
    public bool showPopulationOutputs = true;


    void Start()
    {
        populationOutputLabel = GameObject.Find("Debug").GetComponent<TMP_Text>();

        if (!GetComponent<XORTrainer>().useGeneticAlgorithm)
        {
            network = GetComponent<MLPNetwork>();
            if (network == null || network.inputPerceptrons == null || network.inputPerceptrons.Count == 0)
            {
                Invoke("DrawNetwork", 0.5f);
            }
            else
            {
                DrawNetwork();
            }
        }
    }

    void DrawNetwork()
    {
        ClearVisualization();
        basePosition = this.transform.position;
        neuronPositions.Clear();

        float layerX = 0f;

        DrawLayer(network.inputPerceptrons, layerX, Color.green);
        layerX += horizontalLayerSpacing;

        foreach (var hiddenLayer in network.hiddenLayers)
        {
            DrawLayer(hiddenLayer, layerX, Color.white);
            layerX += horizontalLayerSpacing;
        }

        DrawLayer(network.outputPerceptrons, layerX, Color.blue);

        DrawConnections();
    }
    public void ShowPopulationOutputs(List<MLPNetwork> population)
    {
        System.Text.StringBuilder sb = new System.Text.StringBuilder();
        sb.AppendLine("Population Outputs:");

        for (int i = 0; i < population.Count; i++)
        {
            float output = population[i].lastOutput;
            sb.AppendLine($"[{i}] {output:F3}");
        }
        populationOutputLabel.text = sb.ToString();
    }

    void CreateOutputLabel(Vector3 position, float value)
    {
        GameObject textObj = new GameObject("OutputLabel");
        textObj.transform.position = position;
        var textMesh = textObj.AddComponent<TextMesh>();
        textMesh.text = $"Out: {value:F2}";
        textMesh.fontSize = 40;
        textMesh.characterSize = 0.1f;
        textMesh.color = Color.yellow;
    }

    Vector3 GetPerceptronPosition(Perceptron p)
    {
        return neuronPositions.TryGetValue(p, out Vector3 pos) ? pos : Vector3.zero;
    }

    Dictionary<Perceptron, Vector3> neuronPositions = new Dictionary<Perceptron, Vector3>();

    LineRenderer CreateConnectionLine(Vector3 start, Vector3 end, float weight)
    {
        GameObject lineObj = new GameObject("Connection");
        lineObj.transform.parent = this.transform;
        LineRenderer lr = lineObj.AddComponent<LineRenderer>();
        lr.positionCount = 2;
        lr.SetPosition(0, start);
        lr.SetPosition(1, end);
        lr.startWidth = lineThickness;
        lr.endWidth = lineThickness;

        float intensity = Mathf.Clamp01(Mathf.Abs(weight));
        Color color = Color.Lerp(Color.gray, Color.red, intensity);
        lr.material = new Material(Shader.Find("Sprites/Default"));
        lr.startColor = color;
        lr.endColor = color;

        return lr;
    }

    float GetLayerCenterY(int count)
    {
        return -((count - 1) * verticalNeuronSpacing) / 2f;
    }

    void ClearVisualization()
    {
        foreach (var obj in neurons) Destroy(obj);
        foreach (var line in connections) Destroy(line.gameObject);
        neurons.Clear();
        connections.Clear();
        neuronPositions.Clear();
    }

    void RegisterNeuronPosition(Perceptron p, Vector3 pos)
    {
        if (!neuronPositions.ContainsKey(p))
        {
            neuronPositions[p] = pos;
        }
    }

    void DrawLayer(List<Perceptron> layer, float x, Color neuronColor)
    {
        float centerY = GetLayerCenterY(layer.Count);
        for (int i = 0; i < layer.Count; i++)
        {
            float y = centerY - i * verticalNeuronSpacing;
            Vector3 position = basePosition + new Vector3(x, y, 0);
            GameObject neuron = Instantiate(neuronPrefab, position, Quaternion.identity, this.transform);
            neuron.GetComponent<Renderer>().material.color = neuronColor;
            neurons.Add(neuron);
            RegisterNeuronPosition(layer[i], position);
        }
    }

    void DrawConnections()
    {
        foreach (var kvp in neuronPositions)
        {
            Perceptron target = kvp.Key;
            Vector3 targetPos = kvp.Value;

            if (target.inputs != null)
            {
                foreach (var input in target.inputs)
                {
                    if (input.inputPerceptron != null && neuronPositions.ContainsKey(input.inputPerceptron))
                    {
                        Vector3 start = neuronPositions[input.inputPerceptron];
                        LineRenderer lr = CreateConnectionLine(start, targetPos, input.weight);
                        connections.Add(lr);
                    }
                }
            }
        }
    }

    public void UpdateConnectionColors()
    {
        foreach (var line in connections)
        {
            if (line == null) continue;

            Vector3 start = line.GetPosition(0);
            Vector3 end = line.GetPosition(1);

            // On identifie le perceptron d’arrivée à partir de end
            Perceptron target = GetPerceptronAtPosition(end);
            if (target == null || target.inputs == null) continue;

            foreach (var input in target.inputs)
            {
                Vector3 expectedStart = GetPerceptronPosition(input.inputPerceptron);
                if (Vector3.Distance(start, expectedStart) < 0.01f)
                {
                    float intensity = Mathf.Clamp01(Mathf.Abs(input.weight));
                    Color color = Color.Lerp(Color.gray, Color.red, intensity);
                    line.startColor = color;
                    line.endColor = color;
                    break;
                }
            }
        }
    }

    Perceptron GetPerceptronAtPosition(Vector3 pos)
    {
        foreach (var kvp in neuronPositions)
        {
            if (Vector3.Distance(kvp.Value, pos) < 0.01f)
                return kvp.Key;
        }
        return null;
    }
}
