### Model
I try to implement some basic NLP models including word2vec, transformer, etc.


### Some Pytorch Features, which I encounter during coding

#### Broadcast mechanism
Broadcast is for computing two tensors with different dimensions.
Broadcastable: tensor has at least one dimension, iterate for each dimension, if it one of them is none or is one or equal.

#### Function

torch.eq: element-wise compare, return bool value.

torch.expand: expand one dimension to larger size, input is the desired size