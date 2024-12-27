# Code#1:
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print(arr)
print(type(arr))
# Create a 0-D array with value 42
arr = np.array(42)
print(arr)
# Create a 1-D array containing the values 1,2,3,4,5:
arr = np.array([1, 2, 3, 4, 5])
print(arr)
# Create a 2-D array containing two arrays with the values 1,2,3 and 4,5,6:
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr)
# Create a 3-D array with two 2-D arrays, both containing two arrays with the values 1,2,3 and 4,5,6:
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
print(arr)

# Code#2:

arr = np.array([1, 2, 3, 4])
print(arr[1])
print(arr[2] + arr[3])
# Access 2D array:
# Access the element on the first row, second column:
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print("2nd element on 1st row: ", arr[0, 1])
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print("5th element on 2nd row: ", arr[1, 4])
# Access 3d Array:
# Access the third element of the second array of the first array:
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(arr[0, 1, 2])
# Negative Indexing:
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print("Last element from 2nd dim: ", arr[1, -1])

# Code3:
# Slicing arrays:
# Slicing in python means taking elements from one given index to another given index.
# We pass slice instead of index like this: [start:end].
# We can also define the step, like this: [start:end:step]
# import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])
# Slice elements from index 1 to index 5 from the following array:
print(arr[1:5])
# Slice elements from index 4 to the end of the array:
print(arr[4:])
# Slice elements from the beginning to index 4 (not included):
print(arr[:4])
# Negative Slicing:
# Slice from the index 3 from the end to index 1 from the end:
print(arr[-3:-1])
# STEP
# Use the step value to determine the step of the slicing:
# Return every other element from index 1 to index 5:
print(arr[1:5:2])
# Return every other element from the entire array:
print(arr[::2])
# Slicing 2-D Arrays
# From the second element, slice elements from index 1 to index 4 (not included):
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(arr[1, 1:4])
# From both elements, return index 2:
print(arr[0:2, 2])
# From both elements, slice index 1 to index 4 (not included), this will return a 2-D array:
print(arr[0:2, 1:4])

# Code4:
# Checking the Data Type of an Array
# The NumPy array object has a property called dtype that returns the data type of the array:
# Get the data type of an array object:
arr = np.array([1, 2, 3, 4])
print(arr.dtype)
# Get the data type of an array containing strings:
arr = np.array(["apple", "banana", "cherry"])
print(arr.dtype)
# Iterating Arrays
# Iterating means going through elements one by one.
# As we deal with multi-dimensional arrays in numpy, we can do this using basic for loop of python.
# If we iterate on a 1-D array it will go through each element one by one.
arr = np.array([1, 2, 3])
for x in arr:
    print(x)
# Iterate on each scalar element of the 2-D array:
arr = np.array([[1, 2, 3], [4, 5, 6]])
for x in arr:
    for y in x:
        print(y)
# Iterate on the elements of the following 3-D array:
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
for x in arr:
    print(x)
# To return the actual values, the scalars, we have to iterate the arrays in each dimension.
# Iterate down to the scalars:
for x in arr:
    for y in x:
        for z in y:
            print(z)


#

arr1 = np.array([1, 3, 5, 7, 9, 11])
arr2 = np.array([0, 2, 4, 6, 8, 10])
sum_arr1_arr2 = arr1 + arr2
print("Sum of arr1 and arr2 =>", sum_arr1_arr2)

prod_sum_arr1_arr2 = sum_arr1_arr2 * 4
print("Product of summed array with 4 =>", prod_sum_arr1_arr2)

reshaped_2D_prod_array = prod_sum_arr1_arr2.reshape((2, 3))
print("Reshaped (2D) array =>", reshaped_2D_prod_array)

type_changed_array_double = reshaped_2D_prod_array.astype(np.double)
print("Type (double) array =>", type_changed_array_double)

array_with_gaps_of_2 = np.arange(0, 101, 2)
print("Array with gap of 2 from 0 to 100 =>", array_with_gaps_of_2)

array_with_values = np.array([1, 2, 33, 44, 10])
array_with_duplicate_values = np.array([1, 4, 33, 44, 56])
duplicate_values_indexes = np.where(array_with_values == array_with_duplicate_values)[0]
print("Duplicated values indexes =>", duplicate_values_indexes)
