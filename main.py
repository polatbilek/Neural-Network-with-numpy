from layer import Layer


L1 = Layer(in_size=10, out_size=20, activation_func="tanh", initialization="normal", batch_size=2)


print(L1.forward([[1,1,1,1,-1,1,1,1,1,1],[1,2,1,2,2,2,1,2,3,1]]))

