def solve_crossword(grid, words):
    if not grid:
        return None

    rows, cols = len(grid), len(grid[0])

    def is_valid(word, r, c, dr, dc):
        for i in range(len(word)):
            nr, nc = r + i * dr, c + i * dc
            if not (0 <= nr < rows and 0 <= nc < cols and (grid[nr][nc] == '_' or grid[nr][nc] == word[i])):
                return False
        return True

    def place_word(word, r, c, dr, dc):
        for i in range(len(word)):
            grid[r + i * dr][c + i * dc] = word[i]

    def remove_word(word, r, c, dr, dc):
        for i in range(len(word)):
            grid[r + i * dr][c + i * dc] = '_'

    def solve_util():
        for r in range(rows):
            for c in range(cols):
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    if grid[r][c] == '_' or grid[r][c] == words[0][0]:
                        for word in words:
                            if is_valid(word, r, c, dr, dc):
                                place_word(word, r, c, dr, dc)
                                if solve_util():
                                    return True
                                remove_word(word, r, c, dr, dc)
                        return False
        return True

    if solve_util():
        return grid
    return None

# Example crossword puzzle
grid = [
    ['_', '_', '_', '_', '_', '_', '_'],
    ['_', '_', '_', '_', '_', '_', '_'],
    ['_', '_', '_', '_', '_', '_', '_'],
    ['_', '_', '_', '_', '_', '_', '_'],
    ['_', '_', '_', '_', '_', '_', '_']
]

words = ['apple', 'banana', 'cherry', 'date', 'elderberry', 'fig']

solution = solve_crossword(grid, words)

if solution:
    for row in solution:
        print(' '.join(row))
else:
    print("No solution found.")