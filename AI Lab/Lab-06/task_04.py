from itertools import permutations


class Solver:
    def __init__(self) -> None:
        self.table = {}

    def test_equality(self, s1, s2, s3):
        current, target = 0, 0
        index, log_base = 0, 1
        carry = 0
        while index < len(s1) and index < len(s2):
            s = (
                self.table[s1[len(s1) - index - 1]]
                + self.table[s2[len(s1) - index - 1]]
                + carry
            )
            value = s % 10
            current += value * log_base
            carry = s // 10
            log_base *= 10
            index += 1
        while index < len(s1):
            s = self.table[s1[len(s1) - index - 1]] + carry
            value = s % 10
            current += value * log_base
            carry = s // 10
            log_base *= 10
            index += 1
        while index < len(s2):
            s = self.table[s2[len(s2) - index - 1]] + carry
            value = s % 10
            current += value * log_base
            carry = s // 10
            log_base *= 10
            index += 1
        current += carry * log_base

        index, log_base = 0, 1
        while index < len(s3):
            s = self.table[s3[len(s3) - index - 1]]
            target += s * log_base
            log_base *= 10
            index += 1

        # print(f"Current = {current}, Target = {target}")
        if current == target:
            return True
        else:
            return False

    def solve(self, s1, s2, s3):
        for c in s1:
            if c not in self.table:
                self.table[c] = -1
        for c in s2:
            if c not in self.table:
                self.table[c] = -1
        for c in s3:
            if c not in self.table:
                self.table[c] = -1

        numbers = list(range(1, 10))
        possible_answers = permutations(numbers)
        for answer in possible_answers:
            for key, value in zip(self.table.keys(), answer):
                self.table[key] = value
            if self.test_equality(s1, s2, s3):
                return True
        return False


if __name__ == "__main__":
    solver = Solver()
    s1, s2, s3 = "BASE", "BALL", "GAMES"
    if solver.solve(s1, s2, s3):
        for k, v in solver.table.items():
            print(f"{k} = {v}")
    else:
        print("No solution exists for given cryptarithmetic problem.")
