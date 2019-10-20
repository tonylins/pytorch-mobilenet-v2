# A PyTorch implementation of MobileNetV2

This is a PyTorch implementation of MobileNetV2 architecture as described in the paper [Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation](https://arxiv.org/pdf/1801.04381).

<u>**[NEW]** Add the code to automatically download the pre-trained weights.</u>

## Training Recipe

Recently I have figured out a good training setting:

1. number of epochs: 150
2. learning rate schedule: cosine learning rate, initial lr=0.05
3. weight decay: 4e-5
4. remove dropout

You should get >72% top-1 accuracy with this training recipe!

## Accuracy & Statistics

Here is a comparison of statistics against the official TensorFlow [implementation](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet).

|             | FLOPs     | Parameters | Top1-acc  | Pretrained Model                                             |
| ----------- | --------- | ---------- | --------- | ------------------------------------------------------------ |
| Official TF | 300 M     | 3.47 M     | 71.8%     | -                                                            |
| Ours        | 300.775 M | 3.471 M    | **71.8%** | [[google drive](https://drive.google.com/open?id=1jlto6HRVD3ipNkAl1lNhDbkBp7HylaqR)] |

## Usage

To use the pretrained model, run

```python
from MobileNetV2 import mobilenet_v2

net = mobilenet_v2(pretrained=True)
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