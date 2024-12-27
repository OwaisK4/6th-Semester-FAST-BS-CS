def is_safe(region, color, coloring, neighbors):
    for neighbor in neighbors[region]:
        if neighbor in coloring and coloring[neighbor] == color:
            return False
    return True

def solve_map_coloring(regions, neighbors, colors):
    coloring = {}
    if map_coloring_util(regions, neighbors, colors, coloring):
        return coloring
    return None

def map_coloring_util(regions, neighbors, colors, coloring):
    if len(coloring) == len(regions):
        return True

    region = regions[len(coloring)]
    for color in colors:
        if is_safe(region, color, coloring, neighbors):
            coloring[region] = color
            if map_coloring_util(regions, neighbors, colors, coloring):
                return True
            del coloring[region]

    return False

# Example map with regions and their neighbors
regions = ['WA', 'NT', 'SA', 'QL', 'NSW', 'VIC', 'TAS']
neighbors = {
    'WA': ['NT', 'SA'],
    'NT': ['WA', 'SA', 'QL'],
    'SA': ['WA', 'NT', 'QL', 'NSW', 'VIC'],
    'QL': ['NT', 'SA', 'NSW'],
    'NSW': ['SA', 'QL', 'VIC'],
    'VIC': ['SA', 'NSW', 'TAS'],
    'TAS': ['VIC']
}

colors = ['red', 'green', 'blue']

solution = solve_map_coloring(regions, neighbors, colors)

if solution:
    for region, color in solution.items():
        print(f"{region}: {color}")
else:
    print("No solution found.")