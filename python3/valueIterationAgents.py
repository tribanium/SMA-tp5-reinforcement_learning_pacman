# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from sys import getswitchinterval
import mdp, util

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A ValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs value iteration
    for a given number of iterations using the supplied
    discount factor (gamma).
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your value iteration agent should take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.

        Some useful mdp methods you will use:
            mdp.getStates()
            mdp.getPossibleActions(state)
            mdp.getTransitionStatesAndProbs(state, action)
            mdp.getReward(state, action, nextState)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        "*** YOUR CODE HERE ***"
        # on itère pour actualiser les valeurs
        for iter in range(iterations):
            temp_values = util.Counter()
            for state in mdp.getStates():
                action, score = self.getPolicy(state, get_score=True)
                if not score:
                    score = 0
                temp_values[state] = score
            for state in temp_values:
                self.values[state] = temp_values[state]

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def getQValue(self, state, action):
        """
        The q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that value iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """
        "*** YOUR CODE HERE ***"
        probs_and_states = self.mdp.getTransitionStatesAndProbs(state, action)

        # somme de la q-valeur
        S = 0

        for destination, proba in probs_and_states:
            value = self.getValue(destination)
            transition_reward = self.mdp.getReward(state, action, destination)
            S += proba * (self.discount * value + transition_reward)
            return S

    def getPolicy(self, state, get_score=False):
        """
        The policy is the best action in the given state
        according to the values computed by value iteration.
        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        best_action = None
        best_score = None

        if self.mdp.isTerminal(state):
            if get_score:
                return None, 0
            return None

        for action in self.mdp.getPossibleActions(state):
            Qvalue = self.getQValue(state, action)

            if not best_score:
                best_score = Qvalue
                best_action = action

            if Qvalue > best_score:
                best_action = action
                best_score = Qvalue

        if get_score:
            return best_action, best_score
        return best_action

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.getPolicy(state)
