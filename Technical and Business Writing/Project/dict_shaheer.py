# from collections import defaultdict

top_level = {}

for i in range(10):
    nested = {}
    for j in range(10):
        nested[j] = j
    top_level[i] = nested

print(top_level)