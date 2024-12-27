class NQueenSolver():
    def __init__(self, n) -> None:
        self.n = n
        self.board = [[0 for i in range(n)] for j in range(n)]
    
    def isSafe(self, x, y) -> bool:
        for i in range(x):
            if self.board[i][y] == 1:
                return False
        i, j = x, y
        while i >= 0 and j >= 0:
            if self.board[i][j] == 1:
                return False
            i -= 1
            j -= 1
        i, j = x, y
        while i >= 0 and j < self.n:
            if self.board[i][j] == 1:
                return False
            i -= 1
            j += 1
        return True

    def solve(self, i) -> bool:
        if (i >= self.n):
            for row in self.board:
                print(row)
            return True
        for j in range(0, n):
            if self.isSafe(i, j):
                self.board[i][j] = 1
                if self.solve(i + 1):
                    return True
                self.board[i][j] = 0
        return False

if __name__ == "__main__":
    n = int(input("Enter value of N: "))
    while n < 4 or n > 8:
        print("Invalid input. N must be in the range 4 <= N <= 8.")
        n = int(input("Enter value of N: "))
    solver = NQueenSolver(n)
    solver.solve(0)