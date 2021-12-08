# featureExtractors.py
# --------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"Feature extractors for Pacman game states"

from game import Directions, Actions
import util


class FeatureExtractor:
    def getFeatures(self, state, action):
        """
        Returns a dict from features to counts
        Usually, the count will just be 1.0 for
        indicator functions.
        """
        util.raiseNotDefined()


class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state, action)] = 1.0
        return feats


def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist + 1))
    # no food found
    return None


class SimpleExtractor(FeatureExtractor):
    """
    Returns features for  Pacman
    """

    def getFeatures(self, state, action):
        "*** YOUR CODE HERE ***"

        features = util.Counter()
        features["bias"] = 1.0

        # All states are located in the GameState class of pacman.py
        ghosts_positions = state.getGhostPositions()
        food_positions = state.getFood()
        walls = state.getWalls()
        next_state = state.generatePacmanSuccessor(action)
        next_pacman_position = next_state.getPacmanPosition()

        # Number of ghosts that can reach pacman in one step
        for ghost_position in ghosts_positions:
            possible_ghost_next_positions_list = Actions.getLegalNeighbors(
                ghost_position, walls
            )
            for possible_ghost_next_position in possible_ghost_next_positions_list:
                if possible_ghost_next_position == next_pacman_position:
                    features["nb_ghosts_next_step"] += 1

        # Check if there is food and no ghost on the next step
        if (
            food_positions[next_pacman_position[0]][next_pacman_position[1]]
            and not features["nb_ghosts_next_step"]
        ):
            features["food_next_step"] = 1.0

        # Computes the distance to the next pac-dot
        dist = closestFood(next_pacman_position, food_positions, walls)
        if dist:
            features["food_distance"] = dist / (walls.width * walls.height)

        return features
