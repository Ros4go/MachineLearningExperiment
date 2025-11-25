# Machine Learning Experiment
Il y a plusieurs scène dispo pour tester chaque situation : 

-XOR + Backpropagation + Sigmoide 
-XOR + Backpropagation + TanH 
-XOR + Backpropagation + ReLU : celle ci ne marche pas à 100% parfois elle se stop a 0 et il faut relancer, je dirais que 2 fois sur 4 ca fait ca

-Enemy AI + Backpropagation + Sigmoide 
-Enemy AI + Backpropagation + TanH
-Enemy AI + Backpropagation + ReLU

-Enemy AI + Genetic + Sigmoide : résultat moins fiable que les deux autres
-Enemy AI + Genetic + TanH
-Enemy AI + Genetic + ReLU

## Features en plus :
- Affichage d'un network pour les backpropagation
- Affichage d'un échantillon de resultat pour la genetic
- Les networks ont plusieurs couche "hiden layer"
- Gain reglable en fonction de la fonction de threshold choisi (+ dérivés)
- Fonctions de threshold pour les perceptrons d'input et d'output différenciés des fonctions de threshold pour les hiden layers

![Enemy Exemple](Assets/Ressource/HighEnemyStrenght.png)
![Enemy Exemple](Assets/Ressource/LowEnemyStrenght.png)

# Après le problème de TMP_PRO
J'ai mis en OLD_ les anciennes scènes et j'ai fait la scène FINAL
Dans l'éditeur on peut tweak les parametres de "AI Trainer" : 
- Threshold Input Output Type (ReLU, Sigmoid ou TanH) qui change les threshold type des perceptrons d'input et d'output
- Threshold Type (ReLU, Sigmoid ou TanH) qui change le Threshold Type pour les perceptrons des hiddens layers
- Training set Name (XOR ou AND ou RESULT
- Perceptron number : le nombre de perceptron sur chaque hidden layer
- Use genetic algo
- Threshold Type Genetic : le Threshold Type de l'algo génétique


