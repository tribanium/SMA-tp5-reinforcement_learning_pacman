# Apprentissage par renforcement

L’objectif du TP est d’implémenter et de tester dans différents environnements un algorithme d’apprentissage par renforcement : le Q-learning.

L'implémentation sera effectuée en Python.

## 1. Agent Q-Learning tabulaire

### 1.1. Agent avec exploration manuelle

#### *Questions 1, 2*
Nous allons d'abord tester l'agent avec une exploration manuelle, afin d'observer l'actualisation des Q-valeurs.
Nous complétons la classe `QLearningAgent` (notamment les méthodes `__init__, getQValue, getValue, getPolicy`) afin d'actualiser les Q-valeurs sur les cases. Nous exécutons l'agent avec la commande `python gridworld.py -a q -m -k 5`.

<img src="./screenshots/qlearning-manuel.png" height="500" />

Après quelques itérations, nous obtenons le résultat suivant. Nous avons fait en sorte que `getValue` et `getPolicy` accèdent aux Q-valeurs en appelant `getQValue` uniquement.



### 1.2. Agent avec stratégie d'exploration
Nous allons désormais implémenter la stratégie *greedy* de l'agent dans `getAction`. L'idée ici est d'avoir une politique qui n'est plus figée, car l'agent la modifie par essai/erreur. L'agent va donc à chaque état choisir l'action de plus grande valeur avec une probabilité `1 - epsilon` (exploitation), et choisir une action au hasard avec une probabilité `epsilon` (exploration).

Nous prenons ici `epsilon = 0.3`.

<img src ="./screenshots/epsilon-greedy.png" height="500" />

`python gridworld.py -a q -k 500`

Nous pouvons comparer la politique (les Q-valeurs) obtenue par stratégie *epsilon-greedy* (à droite) après 500 épisodes de l'agent à celle obtenue lors de la planification par MDP (à gauche) après 100 épisodes de value iteration. Nous observons que les ordres de grandeur sont sensiblement similaires pour un nombre suffisant d'itérations. Nous observons cependant des différences notamment au niveau de l'état absorbant de récompense -1 puisque l'agent, malgré les phases d'exploration, suivra la politique optimale 70% du temps.

### 1.3. Nouvel environnement : le robot "crawler"

On considère maintenant un robot composé d’un bras en deux parties. La partie haute du bras possède 4 positions possibles, et la partie basse 6 positions. 4 actions sont possibles correspondant au fait de monter ou descendre d’une position l’une des deux parties du bras. La récompense de l'agent est définie comme la distance parcourue entre deux pasd de temps.

Nous exécutons le crawler afin d'observer l'apprentissage : `python crawler.py`

<img src="./screenshots/crawler.gif" height="400" />

Nous avons la possibilité de modifier en temps réel les différents paramètres d'apprentissage : `discount, epsilon, learning rate`. Nous constatons par exemple qu'augmenter epsilon au début de l'apprentissage permet au robot de mieux cerner son environnement, et en le baissant au fur et à mesure, il suit de plus en plus précisément la politique optimale déterminée.

## 2. Jouons à Pac-Man

On souhaite maintenant avoir un agent qui apprend par renforcement à jouer au jeu de Pacman.

### 2.1. Jeu Pac-Man

Nous testons dans un premier temps notre performance sur Pac-Man `python pacman.py -l smallClassic`.

Force est de constater que nous sommes assez mauvais.

### 2.2. Q-Learning tabulaire pour le jeu Pacman : comment définir les états du MDP ?

### 2.3. Q-Learning et généralisation pour le jeu Pacman
