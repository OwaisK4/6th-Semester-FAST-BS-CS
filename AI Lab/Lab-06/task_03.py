class NQueenCSPSolver:
    def __init__(self) -> None:
        self.n = 4
        self.board = [[0 for i in range(4)] for j in range(4)]

    def isMarked(self, x, y):
        return True if self.board[x][y] > 0 else False

    def print(self):
        for i in self.board:
            for j in i:
                if j == "Q":
                    print(f"{j} ", end="")
                else:
                    print(f"0 ", end="")
            print()
        print()

    # If behaviour is set to true, function will mark cells as invalid
    # Else, will unmark them.
    def mark_cell(self, x_coord, y_coord, behaviour=True):
        self.board[x_coord][y_coord] = "Q" if behaviour else 0
        i = x_coord + 1
        j = y_coord
        while i < self.n:
            if behaviour:
                self.board[i][j] += 1
            else:
                self.board[i][j] -= 1
            i += 1

        i = x_coord
        j = y_coord + 1
        while j < self.n:
            if behaviour:
                self.board[i][j] += 1
            else:
                self.board[i][j] -= 1
            j += 1

        i = x_coord + 1
        j = y_coord + 1
        while i < self.n and j < self.n:
            if behaviour:
                self.board[i][j] += 1
            else:
                self.board[i][j] -= 1
            i += 1
            j += 1

        i = x_coord - 1
        j = y_coord + 1
        while i >= 0 and j < self.n:
            if behaviour:
                self.board[i][j] += 1
            else:
                self.board[i][j] -= 1
            i -= 1
            j += 1

    def Backtrack(self, index):
        self.print()
        if index >= self.n:
            return True
        for i in range(0, self.n):
            if not self.isMarked(i, index):
                self.mark_cell(i, index)
                if self.Backtrack(index + 1):
                    return True
                self.mark_cell(i, index, behaviour=False)
        return False

    def solve(self):
        if self.Backtrack(0):
            print("Final solution: ")
            self.print()
        else:
            print(f"No solution exists for {self.n}x{self.n} chess board")


if __name__ == "__main__":
    solver = NQueenCSPSolver()
    solver.solve()
