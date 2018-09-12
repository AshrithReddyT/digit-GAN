import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from model import generator,discriminator
import os, time, shutil
import matplotlib.pyplot as plt
from PIL import Image


isTraining = False
model = 'new_model'

isTraining = tf.placeholder(dtype=tf.bool)
x = tf.placeholder(tf.float32, shape=(None, 64, 64, 1))
z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))


g=generator(z,training=isTraining)

d_real ,d_real_logits= discriminator(x, training = False)
d_fake , d_fake_logits = discriminator(g, training = False ,reuse = True)


d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits, labels=tf.ones_like(d_real_logits)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.zeros_like(d_fake_logits)))
d_loss = d_loss_real + d_loss_fake
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.ones_like(d_fake_logits)))


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

saver = tf.train.Saver()


try:
    saver.restore(sess, './'.join(['models', model , model]))
except:
    print('Model coult not be restored. Exiting.')
    exit()

test_z = np.random.normal(0, 1, (9, 1, 1, 100))
test_images = sess.run(g, {z: test_z, isTraining: False})
    
dir = './results/'
if not os.path.isdir(dir):
    os.mkdir(dir)

width=320
height=320
cols = 3
rows = 3

for i in range(9):
    data  = np.reshape(test_images[i], (64, 64))
    fig=plt.imshow(data ,cmap='gray', interpolation='nearest')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(dir+'{0}.jpg'.format(i), bbox_inches='tight',pad_inches = 0)
    thumbnail_width = 64
thumbnail_height = 64
time.sleep(5)
size = thumbnail_width, thumbnail_height
new_im = Image.new('RGB', (width, height),(255,255,255,0))
i = 0
ims = []
for p in range(9):
    im = Image.open('./results/'+str(p) + '.jpg')
    img = im.resize(size, Image.ANTIALIAS)
    img.save('./results/'+ str(p) + '.jpg')
    im = Image.open('./results/'+ str(p) + '.jpg')
    ims.append(im)
    i += 1
i = 0
x = 32
y = 32
for col in range(cols):
    for row in range(rows):
        new_im.paste(ims[i], (x, y))
        y += thumbnail_height + 32
        i += 1
    x += thumbnail_width + 32
    y = 32
name = './results/digits.jpg'
new_im.save(name)
for p in range(9):
    os.remove('./results/'+ str(p) + '.jpg')
