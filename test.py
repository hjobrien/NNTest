import numpy as np


# print(np.linspace(0,1,5).reshape(5,1))
# print(np.random.randn(5,1))

f = [np.vstack(tuple([np.linspace(0, 1, num=x) for _ in range(y)])) for y, x in zip(range(3,6), range(1,4))]
for elem in f:
    print("\n")
    print(elem)