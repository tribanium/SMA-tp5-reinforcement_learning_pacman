# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math


class QLearningAgent(ReinforcementAgent):
    """
    Q-Learning Agent

    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.Qvalues = util.Counter()

    def getQValue(self, state, action):
        """
        Returns Q(state,action)
        Should return 0.0 if we never seen
        a state or (state,action) tuple
        """
        "*** YOUR CODE HERE ***"
        return self.Qvalues[(state, action)]

    def getValue(self, state):
        """
        Returns max_action Q(state,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legal_actions = self.getLegalActions(state)

        if not legal_actions:
            return 0.0

        return max(
            [self.getQValue(state, action) for action in self.getLegalActions(state)]
        )

    def getPolicy(self, state):
        """
        Compute the best action to take in a state.  Note that if there
        are no legal actions, which is the case at the terminal state,
        you should return None.
        """
        "*** YOUR CODE HERE ***" ""
        max_value = self.getValue(state)
        legal_actions = self.getLegalActions(state)

        if not legal_actions:
            return None

        for action in legal_actions:
            if self.getQValue(state, action) == max_value:
                return action

    def getAction(self, state):
        """
        Compute the action to take in the current state.  With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise.  Note that if there are
        no legal actions, which is the case at the terminal state, you
        should choose None as the action.

        HINT: You might want to use util.flipCoin(prob)
        HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legal_actions = self.getLegalActions(state)
        action = None

        "*** YOUR CODE HERE ***"
        if not legal_actions:
            return action

        action = self.getPolicy(state)

        if util.flipCoin(self.epsilon):
            action = random.choice(legal_actions)

        return action

    def update(self, state, action, nextState, reward):
        """
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        You should do your Q-Value update here

        NOTE: You should never call this function,
        it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        old_Q = self.getQValue(state, action)
        V = self.getValue(nextState)
        alpha = self.alpha
        gamma = self.discount
        Q = (1 - alpha) * old_Q + alpha * (reward + gamma * V)
        self.Qvalues[(state, action)] = Q


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args["epsilon"] = epsilon
        args["gamma"] = gamma
        args["alpha"] = alpha
        args["numTraining"] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
    ApproximateQLearningAgent

    You should only have to overwrite getQValue
    and update.  All other QLearningAgent functions
    should work as is.
    """

    def __init__(self, extractor="IdentityExtractor", **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)

        # You might want to initialize weights here.
        "*** YOUR CODE HERE ***"
        self.weights = util.Counter()

    def getQValue(self, state, action):
        """
        Should return Q(state,action) = w * featureVector
        where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action)
        Qvalue = 0
        for feature in features.keys():
            Qvalue += self.weights[feature] * features[feature]
        return Qvalue

    def update(self, state, action, nextState, reward):
        """
        Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        next_actions = self.getLegalActions(nextState)
        next_value = None
        for next_action in next_actions:
            next_action_Qvalue = self.getQValue(nextState, next_action)
            if next_value and (next_action_Qvalue > next_value):
                next_value = next_action_Qvalue
        if not next_value:
            next_value = 0

        features = self.featExtractor.getFeatures(state, action)
        alpha = self.alpha
        gamma = self.discount
        update = (reward + (gamma * next_value)) - self.getQValue(state, action)
        self.Qvalues[(state, action)] += alpha * update
        for feature in features.keys():
            self.weights[feature] += alpha * update * features[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
