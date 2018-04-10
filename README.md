# A PyTorch implementation of MobileNetV2

This is a PyTorch implementation of MobileNetV2 architecture as described in the paper [Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation](https://arxiv.org/pdf/1801.04381).

<u>* Special thanks to @wangkuan for providing the model with 71.8% top-1 acc!</u>

## Training & Accuracy

**To train your own model, it is highly recommended to use a Dropout Rate smaller than 0.5 to speed up the training.**

I tried to train the model with RMSprop from scratch as described in the paper, but it does not seem to work. 

I am currently training the model with SGD and keeping other hyper-parameters the same (except that I use batch size 256). I will also try fine-tuning with RMSprop from SGD checkpoint in the future.

The top-1 accuracy on the ImageNet from the paper is **71.7%**. Our current result is **slightly higher**:

| Optimizer     | Epoch | Top1-acc  | Pretrained Model                                             |
| ------------- | ----- | --------- | ------------------------------------------------------------ |
| RMSprop       | -     | -         | -                                                            |
| SGD           | -     | **71.8%** | [[google drive](https://drive.google.com/file/d/1nFZhtKQcw_PeMg8ZZDLdWBcnzqx67hY9/view?usp=sharing)] |
| SGD + RMSprop | TODO  | TODO      | TODO                                                         |

## Usage
To use the pretrained model, run

```python
from MobileNetV2 import MobileNetV2

net = MobileNetV2(n_class=1000)
net = torch.nn.DataParallel(net).cuda()
state_dict = torch.load('mobilenetv2_718.pth.tar')
net.load_state_dict(state_dict)
```

