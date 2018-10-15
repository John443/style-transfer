# style-transfer

## Introduction

This is a project to add styles from famous painting to any photo, which is based on the combination of Gatys' [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576).

## Network

* Use roughly the same transformation network as described in Gatys' paper.
* Use VGG19 instead of VGG16.
* Replace the max pooling layers with average pooling layers as the paper suggests, and disgarded all fully connected layers.
* Compute the style loss at a set of layers rather than just a single layer, then use the weighted sum of style loss at each layer.

## How to use

* Prepare the content image and style image.
* Run `python main.py` to run the program.

## Result

* Content Image
![image](https://github.com/John443/style-transfer/blob/master/images/trojan_shrine.jpg)
* Style Image
![image](https://github.com/John443/style-transfer/blob/master/images/muse.jpg)
* Result
![image](https://github.com/John443/style-transfer/blob/master/images/450.png)

## TODO

* Try to use ResNet instead of VGG19 as the basenet of transfer learning.