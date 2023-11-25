import datasets
import timm
import torch
import numpy
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train().to(device)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X.to(device)).to('cpu')
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test_loop(dataloader, model, loss_fn):
    model.eval().to(device)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.to(device)).to('cpu')
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    return correct


def train_and_test(modelname, dataname):
    print("{:<14} {:<10}".format(modelname, dataname))

    stoplayer = {
        'tf_efficientnet_b5': 'conv_head',
        'convnext_tiny': 'head',
        'resnet50': 'fc',
    }
    ds_data, ds_class = datasets.load(dataname, 224, '/var/tmp/lolyra/artigo', 100)
    c = numpy.zeros(len(ds_data))
    for i in range(len(ds_data)):
        train_dataloader = ds_data[i]['train']
        test_dataloader = ds_data[i]['val']

        model = timm.create_model(modelname, pretrained=True, num_classes = len(ds_class))
        # Disable gradient
        for name, param in model.named_parameters():
            if name.split('.')[0] == stoplayer[modelname]:
                break
            param.requires_grad = False
    
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = nn.CrossEntropyLoss()
        epochs = 10
        for t in range(epochs):
            print(f"Epoch {t+1}",end="\r")
            train_loop(train_dataloader, model, loss_fn, optimizer)
        c[i] = test_loop(test_dataloader, model, loss_fn)
    print("{:.1f} +- {:.1f}".format(100*c.mean(),196*c.std()/numpy.sqrt(len(ds_data))))

if __name__ == "__main__":
    for dataname in ['kth']:
        for modelname in ['convnext_tiny','resnet50']:
            train_and_test(modelname, dataname)
