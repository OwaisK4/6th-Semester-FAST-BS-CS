from constraint import Problem

p = Problem()
p.addVariables(['a','b'], [69, 420])
p.addConstraint(lambda a, b: b == a, ['c', 'b'])

solutions = p.getSolutions()
print(solutions)