import sys


class TSP:
    def __init__(self):
        self.distances = {
            "A": [("B", 20), ("C", 42), ("D", 35)],
            "B": [("A", 20), ("C", 30), ("D", 34)],
            "C": [("A", 42), ("B", 30), ("D", 12)],
            "D": [("A", 35), ("B", 34), ("C", 12)],
        }
        # self.distances = [
        #     [(2, 10), (4, 20), (3, 15)],
        #     [(1, 10), (4, 25), (3, 35)],
        #     [(1, 15), (4, 30), (2, 35)],
        #     [(1, 20), (3, 30), (2, 25)],
        # ]

    def solve(self):
        def calculate_best_solution(src: str, total_cities: int):
            visited = set(src)
            self.best_solution = sys.maxsize
            self.best_path = ""

            def DFS(
                src: str,
                starting_city: str,
                cost: int,
                path: str,
                remaining_cities: int,
            ):
                for dest, dist in self.distances[src]:
                    if remaining_cities <= 0 and dest == starting_city:
                        self.best_solution = min(self.best_solution, cost + dist)
                        self.best_path = path + f" -> {dest}"
                        return
                        return cost + dist
                    if dest not in visited:
                        visited.add(dest)
                        DFS(
                            dest,
                            starting_city,
                            cost + dist,
                            path + f" -> {dest}",
                            remaining_cities - 1,
                        )
                        visited.remove(dest)
                return

            DFS(src, src, 0, src, total_cities - 1)
            return self.best_solution, self.best_path

        cities = self.distances
        min_cost = sys.maxsize
        path: str = ""
        for city in cities:
            solution = calculate_best_solution(city, len(cities))
            if min_cost > solution[0]:
                min_cost = solution[0]
                path = solution[1]
        print(f"Minimum possible cost is: {min_cost}")
        print(f"Path: {path}")


if __name__ == "__main__":
    tsp = TSP()
    tsp.solve()
