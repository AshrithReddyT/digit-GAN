# digit-GAN
**Generating new MNIST digits using a Deep Convolutional Generative Adversarial Network(DCGAN).**

A Generative Adversarial Network has two main Neural Networks.
* Generator
* Discriminator

## Generator

The Generator takes a random input(Noise), and tries to generate the result.
## Discriminator

The Discriminator learns from the actual training data and also the gererated results of the Generator and tries to distinguish real and generated images.
        
## Working of the Model
As the training goes the generator gets better and better at generating realistic outputs while the discriminator finds it harder and harder to distingush real and fake images. 

I ran the model on the so popular MNIST data set and tried generating numbers.Here are the results:

| ![Epoch 1](Epoch%201.jpg) | 
|:--:| 
| *Epoch 1* |

| ![Epoch 14](Epoch%2014.jpg)| 
|:--:| 
| *Epoch 14* |

## Disclaimer
I ran the training on a Nvidia 960M and trust me, it's not that fast(took me 15 minutes per epoch).It might not even fit in your GPU.Just tweak the batch size and it should work fine.By the way the second image was after 14th epoch.I stopped it after that as I already started seeing good enough results and also I was too lazy to wait :p

