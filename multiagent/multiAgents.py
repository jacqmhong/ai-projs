# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()      # pac position after moving
        newFood = successorGameState.getFood()               # remaining food
        newGhostStates = successorGameState.getGhostStates()
        # number moves each ghost will remain scared from pac eating a power pellet
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        if len(newFood.asList()) == 0:
            return 10000

        remainingFood = sum(map(sum, newFood))
        distNearestPellet = min([manhattanDistance(newPos, food) for food in newFood.asList()])

        ghostDists = []
        for i in range(len(newGhostStates)):
            if newScaredTimes[i] == 0:
                ghostDists.append(manhattanDistance(newPos, newGhostStates[i].getPosition()))
        distClosestGhost = min(ghostDists)

        if (len(ghostDists) == 0):
            return 7*currentGameState.getScore() + -2*remainingFood

        if distClosestGhost < 3:
            return -50000

        # dist to closest ghost, dist to closest food, num remaining food
        return 7*successorGameState.getScore() + distClosestGhost + -3*distNearestPellet + -2*remainingFood


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def value(state, depth, agentIndex):
            if agentIndex >= state.getNumAgents():
                agentIndex = 0
                depth -= 1
            # if state is terminal state, return state's utility
            if depth < 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            # max: agent index = 0 means pacman; min: ghosts are >= 1
            if agentIndex == 0:
                return max_value(state, depth, agentIndex)
            if agentIndex >= 1:
                return min_value(state, depth, agentIndex)

        def max_value(state, depth, agentIndex):
            v = -float('inf')
            # for each successor of state, v = max(v, value(successor)); return v
            successorsOfState = []
            for action in state.getLegalActions(agentIndex):
                successorsOfState.append(state.generateSuccessor(agentIndex, action))
            for successor in successorsOfState:
                v = max(v, value(successor, depth, agentIndex+1))
            return v

        def min_value(state, depth, agentIndex):
            v = float('inf')
            # for each successor of state: v = min(v, value(successor)); return v
            successorsOfState = []
            for action in state.getLegalActions(agentIndex):
                successorsOfState.append(state.generateSuccessor(agentIndex, action))
            for successor in successorsOfState:
                v = min(v, value(successor, depth, agentIndex+1))
            return v
            # return min(v, value(successors))

        succs = []
        for action in gameState.getLegalActions(self.index):
            succs.append(action)
        # return max(succs, key = resultValue)
        return max(succs, key = lambda x: value(gameState.generateSuccessor(self.index, x), self.depth-1, 1))

        util.raiseNotDefined()



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def value(state, depth, agentIndex, alpha, beta, action):
            if agentIndex >= state.getNumAgents():
                agentIndex = 0
                depth -= 1
            # if state is terminal state, return state's utility
            if depth < 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), action
            # max: agent index = 0 means pacman; min: ghosts are >= 1
            if agentIndex == 0:
                return max_value(state, depth, agentIndex, alpha, beta)
            if agentIndex >= 1:
                return min_value(state, depth, agentIndex, alpha, beta)

        # def max_value(state, depth, agentIndex, alpha, beta):
        #     v = -float('inf')
        #     # for each successor of state, v = max(v, value(successor)); return v
        #     successorsOfState = []
        #     for action in state.getLegalActions(agentIndex):
        #         successorsOfState.append(state.generateSuccessor(agentIndex, action))
        #     for successor in successorsOfState:
        #         v = max(v, value(successor, depth, agentIndex+1, alpha, beta, action))
        #         if v > beta:
        #             return v
        #         alpha = max(alpha, v)
        #     return v, move

        def max_value(state, depth, agentIndex, alpha, beta):
            v = -float('inf')
            move = Directions.STOP
            # for each successor of state: v = min(v, value(successor)); return v
            for action in state.getLegalActions(agentIndex):
                tmp_v, tmp_move = value(state.generateSuccessor(agentIndex, action), depth, agentIndex+1, alpha, beta, action)
                if tmp_v > v:
                    v = tmp_v
                    move = action
                if v > beta:
                    return v, action
                alpha = max(alpha, v)
            return v, move

        def min_value(state, depth, agentIndex, alpha, beta):
            v = float('inf')
            move = Directions.STOP
            # for each successor of state: v = min(v, value(successor)); return v
            for action in state.getLegalActions(agentIndex):
                tmp_v, tmp_move = value(state.generateSuccessor(agentIndex, action), depth, agentIndex+1, alpha, beta, action)
                if tmp_v < v:
                    v = tmp_v
                    move = action
                if v < alpha:
                    return v, action
                beta = min(beta, v)
            return v, move


        # succs = []
        # for action in gameState.getLegalActions(self.index):
        #     succs.append(action)


        # return max(succs, key = lambda x: value(gameState.generateSuccessor(self.index, x), self.depth-1, 1, -float('inf'), float('inf')))
        # return max_value(gameState, self.depth-1, self.index, -float('inf'), float('inf'))

        # v = -float('inf')
        # alpha = -float('inf')
        # beta = float('inf')
        # move = Directions.STOP
        # for action in gameState.getLegalActions():
        #     temp_v = value(gameState.generateSuccessor(0, action), self.depth, gameState.getNumAgents()-1, -float("inf"), float("inf"))
        #     if temp_v > v:
        #         v = temp_v
        #         move = action
        #     if v > beta:
        #         return v
        #     alpha = max(alpha, v)
        # return move

        return max_value(gameState, self.depth-1, self.index, -float('inf'), float('inf'))[1]

        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def value(state, depth, agentIndex):
            if agentIndex >= state.getNumAgents():
                agentIndex = 0
                depth -= 1
            # if state is terminal state, return state's utility
            if depth < 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            # max: agent index = 0 means pacman; min: ghosts are >= 1
            if agentIndex == 0:
                return max_value(state, depth, agentIndex)
            if agentIndex >= 1:
                return exp_value(state, depth, agentIndex)

        def max_value(state, depth, agentIndex):
            v = -float('inf')
            # for each successor of state, v = max(v, value(successor)); return v
            successorsOfState = []
            for action in state.getLegalActions(agentIndex):
                successorsOfState.append(state.generateSuccessor(agentIndex, action))
            for successor in successorsOfState:
                v = max(v, value(successor, depth, agentIndex+1))
            return v

        def exp_value(state, depth, agentIndex):
            v = 0
            # for each successor of state: p = prob(successor), v += p*value(successor)
            successorsOfState = []
            for action in state.getLegalActions(agentIndex):
                successorsOfState.append(state.generateSuccessor(agentIndex, action))
            for successor in successorsOfState:
                # uniform probability, avg of a ghost node's children
                p = 1 / len(successorsOfState)
                v += p * value(successor, depth, agentIndex+1)
            return v

        succs = []
        for action in gameState.getLegalActions(self.index):
            succs.append(action)
        # return max(succs, key = resultValue)
        return max(succs, key = lambda x: value(gameState.generateSuccessor(self.index, x), self.depth-1, 1))

        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # Useful information you can extract from a GameState (pacman.py)
    # successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = currentGameState.getPacmanPosition()      # pac position after moving
    newFood = currentGameState.getFood()               # remaining food
    newGhostStates = currentGameState.getGhostStates()
    # number moves each ghost will remain scared from pac eating a power pellet
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    if len(newFood.asList()) == 0:
        return 1000000

    remainingFood = sum(map(sum, newFood))
    distNearestPellet = min([manhattanDistance(newPos, food) for food in newFood.asList()])

    ghostDists = []
    for i in range(len(newGhostStates)):
        if newScaredTimes[i] == 0:
            ghostDists.append(manhattanDistance(newPos, newGhostStates[i].getPosition()))

    if (len(ghostDists) == 0):
        return 7*currentGameState.getScore() + -2*remainingFood

    distClosestGhost = min(ghostDists)

    if distClosestGhost < 3:
        return -50000

    newCapsules = currentGameState.getCapsules()
    remainingCapsules = sum(map(sum, newCapsules))

    if newCapsules:
        distNearestCapsule = min([manhattanDistance(newPos, food) for food in newCapsules])
    else:
        distNearestCapsule = 0

    # dist to closest ghost, dist to closest food, num remaining food
    return 7*currentGameState.getScore() + distClosestGhost + -2*distNearestPellet + -2*remainingFood

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
