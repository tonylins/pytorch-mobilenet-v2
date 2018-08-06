# A PyTorch implementation of MobileNetV2

This is a PyTorch implementation of MobileNetV2 architecture as described in the paper [Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation](https://arxiv.org/pdf/1801.04381).

<u>**[NEW]** I fixed a difference in implementation compared to the official TensorFlow model. Please use the new model file and checkpoint!</u>

## Training & Accuracy

I tried to train the model with RMSprop from scratch as described in the paper, but it does not seem to work. I currently train the model with SGD and keeping other hyper-parameters the same (except that I use batch size 256).

Here is a comparison of statistics against the official TensorFlow [implementation](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet).

|             | FLOPs     | Parameters | Top1-acc  | Pretrained Model                                             |
| ----------- | --------- | ---------- | --------- | ------------------------------------------------------------ |
| Official TF | 300 M     | 3.47 M     | 71.8%     | -                                                            |
| Ours        | 300.775 M | 3.471 M    | **71.8%** | [[google drive](https://drive.google.com/open?id=1jlto6HRVD3ipNkAl1lNhDbkBp7HylaqR)] |

## Usage

To use the pretrained model, run

```python
from MobileNetV2 import MobileNetV2

net = MobileNetV2(n_class=1000)
state_dict = torch.load('mobilenetv2.pth.tar') # add map_location='cpu' if no gpu
net.load_state_dict(state_dict)
```

## Data Pre-processing

I used the following code for data pre-processing on ImageNet:

```python
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

input_size = 224
train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=n_worker, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(int(input_size/0.875)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=False,
    num_workers=n_worker, pin_memory=True)
```