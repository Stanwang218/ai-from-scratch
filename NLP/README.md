### Model
I try to implement some basic NLP models including word2vec, transformer, etc.


### Some Pytorch Features, which I encounter during coding

#### Broadcast mechanism
Broadcast is for computing two tensors with different dimensions.
Broadcastable: tensor has at least one dimension, iterate for each dimension, if it one of them is none or is one or equal.

#### Pytorch Function

torch.eq: element-wise compare, return bool value.

torch.Tensor.expand: expand one dimension to larger size, input is the desired size
torch.Tensor.masked_filled_(mask, value): mask tensor with value in the position of true in mask
torch.Tensor.repeat(int/size): similar to np.tile(), eg: the size of tensor is [4], repeat(4, 2) -> the shape becomes (4, 8)
torch.Tensor.contiguous: transform the tensor into tensor with contiguous memory storage

#### Numpy Function
np.triu(array, k): return an upper triangle, k indicates the k-th diagonal, k = 0 indicates the main diagonal 