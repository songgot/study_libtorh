---
title: Load after saving trained model
...

## Load after saving trained model
To store model weights in Pytorch, we need three functions
* torch.save
* torch.load
* torch.nn.Module.load_state_dict : load the parameters of the model by state_dict

### Let's practice with a simple DNN model
#### model training
```
import torch
import torch.nn as nn

x_data = torch.Tensor([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 0],
    [0, 0],
    [0, 1]
])

y_data = torch.LongTensor([
    0,  # etc
    1,  # mammal
    2,  # birds
    0,
    0,
    2
])

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.w1 = nn.Linear(2, 10)
        self.bias1 = torch.zeros([10])

        self.w2 = nn.Linear(10, 3)
        self.bias2 = torch.zeros([3])
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        y = self.w1(x) + self.bias1
        y = self.relu(y)

        y = self.w2(y) + self.bias2
        return y

model = DNN()

criterion = torch.nn.CrossEntropyLoss() #loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    output = model(x_data)

    loss = criterion(output, y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("progress:", epoch, "loss=", loss.item())
```
#### torch.save(object, path)
Used to save the entire model or the state_dict of the model.
* object: model object to save
* path: location to save + file name
```
#orch.save(model, 'model.pt')  # save entire model
torch.save(model.state_dict(), 'model_state_dict.pt')  # Store state_dict of model object
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict()}, 'all.tar')  # store various values and save progress during learning.
```
#### troch.load(path)
Used when loading the entire model or the state_dict of the model.
path: Location to load + file name

```
model = torch.load(PATH + 'model.pt')  # Load the entire model as a whole, class declaration required
```
or
#### torch.nn.Module.load_state_dict(dict)
Using state_dict, initialize parameter values in the model object.
* dict: state_dict object containing the parameter values to be loaded.
```
model.load_state_dict(torch.load(PATH + 'model_state_dict.pt'))  # After loading state_dict, save it to model
```
or
```
checkpoint = torch.load(PATH + 'all.tar')   # load all.tar
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
```
#### re-training with x_data and y_data
```
for epoch in range(1000):
    output = model(x_data)

    loss = criterion(output, y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("progress:", epoch, "loss=", loss.item())
```
