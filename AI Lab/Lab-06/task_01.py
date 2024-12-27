class TicTacToe:
    def __init__(self, board=None):
        self.n = 3
        self.board = None
        if board is None:
            self.board = [["" for i in range(3)] for j in range(3)]
        else:
            self.board = board

    def print(self):
        for i in self.board:
            for j in i:
                if j == "":
                    print(f"  ", end="")
                else:
                    print(f"{j} ", end="")
            print()
        print()

    def Win(self, player):
        if self.board[0][0] == self.board[1][1] == self.board[2][2] == player:
            return True
        if self.board[2][0] == self.board[1][1] == self.board[0][2] == player:
            return True

        if self.board[0][0] == self.board[0][1] == self.board[0][2] == player:
            return True
        if self.board[1][0] == self.board[1][1] == self.board[1][2] == player:
            return True
        if self.board[2][0] == self.board[2][1] == self.board[2][2] == player:
            return True

        if self.board[0][0] == self.board[1][0] == self.board[2][0] == player:
            return True
        if self.board[0][1] == self.board[1][1] == self.board[2][1] == player:
            return True
        if self.board[0][2] == self.board[1][2] == self.board[2][2] == player:
            return True

        return False

    def solve(self):
        open_list = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == "":
                    open_list.append((i, j))
        closed_list = []
        if self.Backtrack("X", open_list, closed_list):
            self.print()
        else:
            print("No solution possible.")

    def Backtrack(self, player, open_list, closed_list):
        self.print()
        if self.Win(player):
            print(f"Player {player} won the game.")
            return True
        if len(open_list) == 0:
            return False
        x, y = open_list.pop()
        closed_list.append((x, y))
        self.board[x][y] = player
        if player == "O":
            if self.Backtrack("X", open_list, closed_list):
                return True
        else:
            if self.Backtrack("O", open_list, closed_list):
                return True
        x, y = closed_list.pop()
        open_list.append((x, y))
        return True


if __name__ == "__main__":
    board = [
        ["O", "", "X"],
        ["X", "", ""],
        ["X", "O", "O"],
    ]
    solver = TicTacToe(board)
    solver.solve()
