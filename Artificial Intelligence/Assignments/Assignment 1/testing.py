def func(n):
    def factorial(n):
        if n < 2:
            return 1
        else:
            return n * factorial(n - 1)

    return factorial(n)


if __name__ == "__main__":
    ans = func(5)
    print(ans)
