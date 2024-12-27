from random import randint

celsius = [randint(0, 100) for i in range(9)]
fahrenheit = [temp * (9 / 5) + 32 for temp in celsius]
average_temp_fahrenheit = sum(fahrenheit) / len(fahrenheit)
print("Average temperature in Fahrenheit:", average_temp_fahrenheit)
