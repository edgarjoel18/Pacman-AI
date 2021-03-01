# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
import heapq

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

# End of SearchProblem class
def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]



def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    # Creating an array of visited nodes
    visitedNodes = []
    currentArray = []
    stack = util.Stack()  # Creating a object of type stack. Note: to self that in python there is no need
    # to use the word new to create an object

    stack.push((problem.getStartState(), currentArray))
    # We have our (location, path)

    while not stack.isEmpty():
        currentNode = stack.pop()
        currentState = currentNode[0]
        path = currentNode[1]
        visitedNodes.append(currentState)
        # We have to check if we reached the goal. If so, return the path that took us to this goal
        goalStateFound = problem.isGoalState(currentState)
        if goalStateFound == True:
            return path
        # else Check if the the currentState has not been visited yet. If not set it as visited
        # and explore its children
        # elif currentNode not in visitedNodes:
        #     visitedNodes.add(currentState)
            # We must expand the child nodes of the current State
            """
            Note to self: This is helpful from searchAgents.py. It describes the function getSuccessors
            For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
            """
        for childNode in problem.getSuccessors(currentState):
            childState = childNode[0]
            childPath = childNode[1]
            if childState not in visitedNodes:
                childPath = path + [childPath]
                stack.push((childState,childPath))



        # for item in problem.getSuccessors(currentState):
        #     if item[0] in visitedNodes:
        #         continue
        #     stack.push((item[0], currentNode[1] + [item[1]]))






    return None
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"


    visitedNodes = []
    # For level order traversal we need a queue data structure
    queue = util.Queue()
    # Same process as DFS rather we are pulling from the front of this data structure rather than the back
    queue.push((problem.getStartState(), []))
    # If we exit this while loop we didnt find the golden state
    visitedNodes.append(problem.getStartState())
    while not queue.isEmpty():
        currentNode = queue.pop() # enqueue the current node
        currentState = currentNode[0]
        # The path includes the cost to a certain state
        currentPath = currentNode[1]
        goalStateFound = problem.isGoalState(currentState)
        if goalStateFound == True:
            return currentPath
        # elif currentNode not in visitedNodes:
        #     visitedNodes.append(currentNode)
        for childNode in problem.getSuccessors(currentState):
            childState = childNode[0]
            childPath = childNode[1]
            if childState not in visitedNodes:
                visitedNodes.append(childNode[0])
                childPath = currentPath + [childPath]
                # Add child nodes to the queue to visit later on in the process
                queue.push((childState, childPath))
    return None
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    # We still need to keep track of our visited nodes
    visitedNodes = []
    # We need to min Heap to extract the min cost path
    minHeap = util.PriorityQueue()
    minHeap.push((problem.getStartState(), [], 0),0)
    # minHeap takes 3 args. Now we have (location, path, currentCost), priority


    while not minHeap.isEmpty():
        # We are always extracting the min path in O(1) time
        currentNode = minHeap.pop()
        # currentNode[2] is our cummulative cost
        currentState = currentNode[0]
        currentPath = currentNode[1]
        currentCost = currentNode[2]
        # We check if this is the cheapest cost Search and if we found our goalNode
        goalNodeFound = problem.isGoalState(currentState)
        if goalNodeFound == True:
            return currentPath
        # else if we haven't visited the currentState mark as visited and traverse its children while adding their cost and
        # adding their path and set it to the minHeap so the minHeap can heapify the cheapest path at the root
        elif currentState not in visitedNodes:
            visitedNodes.append(currentState)
            for childNode in problem.getSuccessors(currentState):
                childState = childNode[0]
                childPath = childNode[1]
                childCost = childNode[2]
                # if the childState has not been visited then we can compute the pathCost from the currentChildCost and from the
                # currentNode cost
                if childState not in visitedNodes:
                    cummulativeCost = currentCost + childCost
                    # We push the total cost path to our minHeap and once we pop. Which is O(1) time
                    # We get the cheapestPath in order to find the goalState
                    minHeap.push((childNode[0], currentNode[1] + [childNode[1]], cummulativeCost), cummulativeCost)

    return None
    # util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    visitedNodes = []
    minHeap = util.PriorityQueue()
    minHeap.push((problem.getStartState(),[],0), 0)

    # A* is a acts like Uniform cost search and Best First Search

    while not minHeap.isEmpty():
        currentNode = minHeap.pop()
        currentState = currentNode[0]
        currentPath = currentNode[1]
        currentCost = currentNode[2]
        # Check if our currentState is the goal state
        isGoalState = problem.isGoalState(currentState)
        if isGoalState == True:
            return currentPath
        # else if currentState has not been visited then set it as visited and traverse it.
        # While traversing to each childNode of the currentState. Compute the backward cost and future cost
        # In other words f(n) = g(n) + h(n) then Push its child nodes on the minHeap.
        # Once we pop from the heap we have an optimal and cheapest solution
        elif currentState not in visitedNodes:
            visitedNodes.append(currentState)
            for childNode in problem.getSuccessors(currentState):
                childState = childNode[0]
                childPath = childNode[1]
                childCost = childNode[2]
                if childState not in visitedNodes:
                    # g(n)
                    backwardCost = childCost + currentCost
                    # f(n) = h(n) + g(n)
                    totalCost = heuristic(childState, problem) + backwardCost
                    minHeap.push((childState, currentPath + [childPath], backwardCost), totalCost)

    return None
    # util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
