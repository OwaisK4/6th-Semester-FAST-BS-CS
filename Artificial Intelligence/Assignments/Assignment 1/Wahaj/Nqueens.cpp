#include <iostream>

using namespace std;
int solutions = 0;
bool isSafe(int **board, int row, int col, int size) {
    if (row < 0 || row >= size || col < 0 || col >= size) {
        return false;
    }
    // left
    for (int i = col - 1; i >= 0; i--) {
        if (board[row][i] == 1) {
            return false;
        }
    }
    // right
    for (int i = col + 1; i < size; i++) {
        if (board[row][i] == 1) {
            return false;
        }
    }
    // up
    for (int i = row - 1; i >= 0; i--) {
        if (board[i][col] == 1) {
            return false;
        }
    }
    // diagonally left upwards
    int r = row - 1;
    int c = col - 1;
    while (r >= 0 && c >= 0) {
        if (board[r][c] == 1) {
            return false;
        }
        r--;
        c--;
    }
    // diagonally right upwards
    r = row - 1;
    c = col + 1;
    while (r >= 0 && col < size) {
        if (board[r][c] == 1) {
            return false;
        }
        r--;
        c++;
    }
    return true;
}
void nQueens(int **board, int row, int col, int total, int current) {
    if (total == current) {
        for (int i = 0; i < total; i++) {
            for (int j = 0; j < total; j++) {
                cout << board[i][j] << " ";
            }
            cout << endl;
        }
        cout << endl;
        solutions++;
        return;
    }
    if (col == total) {
        row++;
        col = 0;
    }
    if (row == total) {
        return;
    }
    if (isSafe(board, row, col, total)) {
        board[row][col] = 1;
        nQueens(board, row, col + 1, total, current + 1);
        board[row][col] = 0;
    }
    nQueens(board, row, col + 1, total, current);
    return;
}
int main() {
    int n;
    cin >> n;
    int **board = new int *[n];
    for (int i = 0; i < n; i++) {
        board[i] = new int[n];
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            board[i][j] = 0;
        }
    }
    nQueens(board, 0, 0, n, 0);

    cout << "No of solutions: " << solutions << endl;
}