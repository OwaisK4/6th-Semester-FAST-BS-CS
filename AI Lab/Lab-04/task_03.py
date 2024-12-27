import sys
from typing import List
from collections import deque

class EightPuzzle:
    def __init__(self, initial: List[List[int]], target: List[List[int]]):
        self.initial = initial
        self.target = target
        assert len(self.initial) == len(self.target)
        self.n = len(self.initial)
    def compare_states(self):
        for i in range(self.n):
            for j in range(self.n):
                if self.initial[i][j] != self.target[i][j]:
                    return False
        return True
    def Solve_DFS(self):
        x, y = -1, -1
        for i in range(self.n):
            for j in range(self.n):
                if self.initial[i][j] == 0:
                    x, y = i, j
                    break
            if (x, y) != (-1, -1):
                break
        print(x, y)
        visited = [[False for i in range(self.n)] for j in range(self.n)]
        print(visited)


if __name__ == "__main__":
    start_state = [ [1, 2, 3],
                    [5, 6, 0],
                    [7, 8, 4] ]
    goal_state = [ [1, 2, 3],
                    [5, 8, 6],
                    [0, 7, 4] ]
    solver = EightPuzzle(start_state, goal_state)
    solver.Solve_DFS()