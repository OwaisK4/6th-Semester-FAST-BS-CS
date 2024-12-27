import heapq

goal_state = (0,1, 2, 3, 4, 5, 6, 7, 8)

moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]  

def manhattan_distance(state):
    distance = 0
    for i in range(3):
        for j in range(3):
            if state[i * 3 + j] != 0:
                target_row = (state[i * 3 + j] - 1) // 3
                target_col = (state[i * 3 + j] - 1) % 3
                distance += abs(i - target_row) + abs(j - target_col)
    return distance

def get_next_states(state):
    zero_index = state.index(0)
    zero_row, zero_col = zero_index // 3, zero_index % 3
    next_states = []
    for move in moves:
        new_row, new_col = zero_row + move[0], zero_col + move[1]
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            next_state = list(state)
            next_state[zero_index], next_state[new_row * 3 + new_col] = next_state[new_row * 3 + new_col], next_state[zero_index]
            next_states.append(tuple(next_state))
    return next_states

def a_star(initial_state):
    open_list = [(manhattan_distance(initial_state), 0, initial_state, [])]
    closed_set = set()

    while open_list:
        _, g_score, current_state, path = heapq.heappop(open_list)
        if current_state == goal_state:
            return path
        if current_state in closed_set:
            continue
        closed_set.add(current_state)
        for next_state in get_next_states(current_state):
            if next_state not in closed_set:
                f_score = g_score + 1 + manhattan_distance(next_state)
                next_path = path + [next_state]
                heapq.heappush(open_list, (f_score, g_score + 1, next_state, next_path))

initial_state = (1, 2, 3, 0, 4, 6, 7, 5, 8)  
steps = a_star(initial_state)
print("Number of steps to solve the puzzle:", steps)
solution_path = a_star(initial_state)
if solution_path:
    print("Steps to solve the puzzle:")
    for i, state in enumerate(solution_path):
        print("Step", i + 1, ":", state)
else:
    print("No solution found!")