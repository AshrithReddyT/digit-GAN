import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from model import generator,discriminator
import os, time, shutil
import matplotlib.pyplot as plt
from PIL import Image

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True ,reshape=[])
np.random.seed(int(time.time()))

test_z = np.random.normal(0, 1, (9, 1, 1, 100))
def test_generator(num_epoch, width=320, height=320, cols = 3,rows = 3):
    test_images = sess.run(g, {z: test_z, isTraining: False})
    
    dir = './results/Epoch {0}/'.format(num_epoch)
    if not os.path.isdir(dir):
        os.mkdir(dir)

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
        im = Image.open('./results/Epoch {0}/'.format(num_epoch) + str(p) + '.jpg')
        img = im.resize(size, Image.ANTIALIAS)
        img.save('./results/Epoch {0}/'.format(num_epoch) + str(p) + '.jpg')
        im = Image.open('./results/Epoch {0}/'.format(num_epoch) + str(p) + '.jpg')
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
    name = './results/Epoch {0}.jpg'.format(num_epoch)
    new_im.save(name)
    shutil.rmtree(dir)

isTraining = tf.placeholder(dtype=tf.bool)

batch_size = 50
learning_rate = 0.0002
epochs = 25

x = tf.placeholder(tf.float32, shape=(None, 64, 64, 1))
z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))


#create Networks
g=generator(z,training=isTraining)

d_real ,d_real_logits= discriminator(x, training = isTraining)
d_fake , d_fake_logits = discriminator(g, training = isTraining ,reuse = True)

#compute Loss
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits, labels=tf.ones_like(d_real_logits)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.zeros_like(d_fake_logits)))
d_loss = d_loss_real + d_loss_fake
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.ones_like(d_fake_logits)))

#Variables
d_training_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
g_training_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]

#optimizer
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    d_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(d_loss, var_list=d_training_vars)
    g_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(g_loss, var_list=g_training_vars)

if not os.path.isdir('results'):
    os.mkdir('results')

#Session Configuration
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

train_images = tf.image.resize_images(mnist.train.images, [64, 64]).eval()
train_images = (train_images - 0.5) / 0.5

#Training
print('Started Training')
training_start_time = time.time()
for epoch in range(epochs):
    g_losses_this_epoch = []
    d_losses_this_epoch = []
    iters = int(mnist.train.num_examples/batch_size)
    epoch_start_time = time.time()
    for iter in range(iters):
        X = train_images[iter*batch_size:(iter+1)*batch_size]
        Z = np.random.normal(0, 1, (batch_size, 1, 1, 100))
        loss_d_this_iter, _ = sess.run([d_loss,d_optimizer], {x: X, z: Z, isTraining: True})
        Z = np.random.normal(0, 1, (batch_size, 1, 1, 100))
        loss_g_this_iter, _ = sess.run([g_loss,g_optimizer], {z: Z, x: X, isTraining: True})
        d_losses_this_epoch.append(loss_d_this_iter)
        g_losses_this_epoch.append(loss_g_this_iter)

    epoch_end_time = time.time()
    total_epoch_time = time.strftime("%H:%M:%S", time.gmtime(int(epoch_end_time - epoch_start_time)))
    print('Epoch :{0}/{0} took time: '.format(str(epoch + 1),str(epochs)) + total_epoch_time +'    Discriminator loss: {0}    Generator loss: {0}' .format(str(np.mean(d_losses_this_epoch)), str(np.mean(g_losses_this_epoch))))
    test_generator(epoch + 1)

training_end_time = time.time()
total_training_time = time.strftime("%H:%M:%S", time.gmtime(int(training_end_time - training_start_time)))

print("Training Complete")
print('Total time taken for training: {0} epochs was '.format(str(epochs)) + total_training_time)

sess.close()
