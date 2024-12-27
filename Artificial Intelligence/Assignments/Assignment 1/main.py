import os, sys
from Search import Search


# Reads the heuristics file stored as csv and returns a dictionary
# containing heuristics data. In case of error, returns string.
def read_heuristics_file(filename: str) -> dict[str, int] | str:
    heuristics = {}
    try:
        with open(filename, "r") as f:
            for i in f:
                try:
                    key, value = i.split(",")
                    value = int(value)
                    heuristics[key] = value
                except:
                    print("Malformed data in heuristics file.")
                    sys.exit()
    except FileNotFoundError:
        print("Heuristics file does not exist.")
        sys.exit()
    return heuristics


# Reads the distances file stored as csv and returns a dictionary
# containing distances data. In case of error, returns string.
def read_distances_file(filename: str) -> dict[str, list[(str, int)]] | str:
    distances = {}
    try:
        with open(filename, "r") as f:
            for i in f:
                try:
                    src, dest, value = i.split(",")
                    value = int(value)
                    distances.setdefault(src, [])
                    distances[src].append((dest, value))
                    distances.setdefault(dest, [])
                    distances[dest].append((src, value))
                    # distances[src] = (dest, value)
                    # distances[dest] = (src, value)
                except:
                    print("Malformed data in distances file.")
                    sys.exit()
    except FileNotFoundError:
        print("Distances file does not exist.")
        sys.exit()
    return distances


# Main driver code
if __name__ == "__main__":
    basepath = os.path.abspath(".")

    heuristics_filename = "heuristics.txt"
    heuristics_filepath = os.path.join(basepath, heuristics_filename)
    heuristics = read_heuristics_file(heuristics_filepath)
    # print(heuristics)

    distances_filename = "distances.txt"
    distances_filepath = os.path.join(basepath, distances_filename)
    distances = read_distances_file(distances_filepath)
    # print(distances)

    possible_sources = set(distances.keys())
    print(possible_sources)

    searcher = Search(distances, heuristics)

    while True:
        src = input("Enter a source location from the above map: ")
        while src not in possible_sources and src != "-1":
            print("Given source does not exist in the map. Try again.\n")
            src = input("Enter a source location from the above map: ")
        if src == "-1":
            break
        dest = input("Enter a destination location from the above map: ")
        while dest not in possible_sources and dest != "-1":
            print("Given source does not exist in the map. Try again.\n")
            dest = input("Enter a destination location from the above map: ")
        if dest == "-1":
            break

        path, cost = searcher.BreadthFirstSearch(src, dest)
        print(f"\nUsing BreadthFirstSearch:")
        print(f"Path = {path}")
        print(f"Cost = {cost}")
        path, cost = searcher.UniformCostSearch(src, dest)
        print(f"\nUsing UniformCostSearch:")
        print(f"Path = {path}")
        print(f"Cost = {cost}")
        path, cost = searcher.GreedyBestFirstSearch(src, dest)
        print(f"\nUsing GreedyBestFirstSearch:")
        print(f"Path = {path}")
        print(f"Cost = {cost} (Heuristics)")
        path, cost = searcher.IterativeDeepeningDepthFirstSearch(src, dest)
        print(f"\nUsing IterativeDeepeningDepthFirstSearch:")
        print(f"Path = {path}")
        print(f"Cost = {cost}\n")
    print("Exiting searcher program.")
