import numpy as np

a = np.array([[[1, 2, 3], [1, 2, 3]],
               [[1, 2, 3], [1, 2, 3]]])


print(a.transpose(2, 0, 1))
print("Original array with shape (2, 2, 3)")
print(a)
print(a.reshape((3, 2, 2)))