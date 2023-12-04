import torch
import torch.nn as nn


criterion = nn.CrossEntropyLoss()

n_class = 4

# There are three loss functions related with Entropy
# CrossEntropyLoss, BCELoss, NLLLoss
# Softmax, LogSoftmax
# LogSoftmax = torch.log(torch.softmax(input_data, dim = 1))
# LogSoftmax is defined in the torch.nn

def validate_cross_entropy():
    y = torch.tensor([0,1,2,3])
    torch.random.manual_seed(1)
    pred_y = torch.rand([4,4])
    
    print(pred_y)
    print(nn.LogSoftmax(dim=1)(pred_y))
    print(torch.log(torch.softmax(pred_y, dim=1)))
    
    pred_y = torch.log(torch.softmax(pred_y, dim=1))
    
    criterion = nn.CrossEntropyLoss()
    
    y_tensor = torch.zeros_like(pred_y)
    ans = 0
    for i in range(pred_y.shape[0]):
        y_tensor[i, y[i]] = 1
        ans += torch.inner(pred_y[i,:],y_tensor[i,:])
        # ans += torch.inner(pred_y[i,:],y_tensor[i,:])
    
    
    print(criterion(pred_y, y))
    print(ans / 4)
    
    
if __name__ == '__main__':
    validate_cross_entropy()