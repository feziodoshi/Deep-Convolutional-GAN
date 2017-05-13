import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
import datetime
import cv2

# collecting dataset for mnist
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/")
total_epoch=200000

# will first create a discrimniator which is a simple convolutional layer
# we will create 2 convolutional layers and 2 feed forward layers
def discriminator(x_image, reuse=False):

    if reuse:
        tf.get_variable_scope().reuse_variables()


    # creating first convolutional layer
    # creating the weight and bias of discriminator layer
    d_1_w=tf.get_variable(name="d_1_w", shape=[5,5,1,32], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.2))
    d_1_b=tf.get_variable(name='d_1_b', shape=[32], initializer=tf.constant_initializer(0))
    d_1=tf.nn.conv2d(input=x_image,filter=d_1_w,strides=[1,1,1,1],padding='SAME')
    d_1=d_1+d_1_b
    d_1=tf.nn.relu(d_1)
    d_1=tf.nn.avg_pool(value=d_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    ##creating the second convolutional layer
    ##creating the weights and bias of the second convolutional layer
    d_2_w=tf.get_variable(name="d_2_w",shape=[5,5,32,64],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.2))
    d_2_b=tf.get_variable(name="d_2_b",shape=[64],initializer=tf.constant_initializer(0))
    d_2=tf.nn.conv2d(input=d_1,filter=d_2_w,strides=[1,1,1,1],padding="SAME")
    d_2=d_2+d_2_b
    d_2=tf.nn.relu(d_2)
    d_2=tf.nn.avg_pool(value=d_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    ##reshaping the input vector to a flat matrix
    d_2=tf.reshape(d_2,[-1,7*7*64])

    ##creating the first feed forward layer
    ##creating weights of this layer
    d_3_w=tf.get_variable(name='d_3_w',shape=[7*7*64,1024],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.2))
    d_3_b=tf.get_variable(name='d_3_b',shape=[1024],initializer=tf.constant_initializer(0))
    d_3=tf.add(tf.matmul(d_2,d_3_w),d_3_b)
    d_3=tf.nn.relu(d_3)

    ##creating second feed forward layer
    ##creating weights and biases of this layer
    d_4_w=tf.get_variable(name='d_4_w',shape=[1024,1],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.2))
    d_4_b=tf.get_variable(name='d_4_b',shape=[1],initializer=tf.constant_initializer(0))
    d_4=tf.add(tf.matmul(d_3,d_4_w),d_4_b)
    return d_4


##lets say we start with zdim=100
def generator(batch_size,zdim):
    ##creating the input which is the noise
    z=tf.truncated_normal(shape=[batch_size,zdim],mean=0,stddev=1,dtype=tf.float32)

    ## deconvolution=converting a flat layer to a 2d layer and then continuing with convolution such that
    ## the number of feature maps reeduce and finally become 1 and you end up with the output dimension of your choice

    ##deconvolving first creating a matrix form from flat
    ##first making for (56*56)
    g_1_w=tf.get_variable(name='g_1_w',shape=[zdim,3136],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_1_b=tf.get_variable(name='g_1_b',shape=[3136],initializer=tf.constant_initializer(0))
    g_1=tf.add(tf.matmul(z,g_1_w),g_1_b)
    g_1=tf.reshape(g_1,[-1,56,56,1])
    g_1=tf.nn.batch_normalization(g_1,mean=0,variance=0.02,variance_epsilon=1e-5,name='bn1',offset=None,scale=None)
    g_1=tf.nn.relu(g_1)

    ##generate 50(zdim/2=100/2) features and just convolve after batch normalization
    ##creating weights and bias for (56*56)
    g_2_w=tf.get_variable(name='g_2_w',shape=[3,3,1,zdim/2],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_2_b=tf.get_variable(name='g_2_b',shape=[zdim/2],initializer=tf.constant_initializer(0))
    g_2=tf.nn.conv2d(input=g_1,filter=g_2_w,strides=[1,2,2,1],padding='SAME')
    g_2=g_2+g_2_b
    g_2 = tf.nn.batch_normalization(g_2,mean=0,variance=0.02,variance_epsilon=1e-5,name='bn2',offset=None,scale=None)
    g_2=tf.nn.relu(g_2)
    g_2=tf.image.resize_images(g_2,[56,56])

    ##generate 25 features and just convolve after batch normalization
    ##creating weights and bias for (56,56)
    g_3_w=tf.get_variable(name='g_3_w',shape=[3,3,zdim/2,zdim/4],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_3_b=tf.get_variable(name='g_3_b',shape=[zdim/4],initializer=tf.constant_initializer(0))
    g_3=tf.nn.conv2d(input=g_2,filter=g_3_w,strides=[1,2,2,1],padding='SAME')
    g_3=g_3+g_3_b
    g_3=tf.nn.batch_normalization(x=g_3,mean=0,variance=0.02,variance_epsilon=1e-5,name='bn3',offset=None,scale=None)
    g_3=tf.nn.relu(g_3)
    g_3=tf.image.resize_images(g_3,[56,56])

    ##generting one feature map of size 28*28
    ## this is the fourth block of the generator and we will generate the final image that will be generated

    g_4_w=tf.get_variable(name='g_4_w',shape=[3,3,zdim/4,1],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_4_b=tf.get_variable(name='g_4_b',shape=[1],initializer=tf.constant_initializer(0))
    g_4=tf.nn.conv2d(input=g_3,filter=g_4_w,strides=[1,2,2,1],padding='SAME')
    g_4=g_4+g_4_b
    ##no need to batch normalize here
    ##the activation that we will be using here is sigmoid to make the output more crisper- more like the probablities
    g_4=tf.sigmoid(g_4)

    return (g_4)

# def write_summary(g_loss,d_loss_real,d_loss_fake,batch_size,z_dim,x,sess):
#     tf.summary.scalar('Generator Loss',g_loss)
#     tf.summary.scalar('Discriminator Real loss',d_loss_real)
#     tf.summary.scalar('Discriminator Fake loss',d_loss_fake)
#
#     ##sanity check for checking how the discriminator performs on the fake images
#     d_eval_fake=tf.reduce_mean(discriminator(generator(batch_size,z_dim)))
#     tf.summary.scalar('Fake Evaluation',d_eval_fake)
#     ##sanity check for checking how the discriminator performs on real images
#     d_eval_real=tf.reduce_mean(discriminator(x))
#
#     ##get a list of all images and show it on tensorboard
#     generated_images=generator(batch_size,z_dim)
#     tf.summary.image('Generated images',generated_images,10)
#     merged=tf.summary.merge_all()
#     logdir="C:/Users/ezio/Desktop/GAN/one/Tensorboard/"
#     writer=tf.summary.FileWriter(logdir,sess.graph)
#     print(logdir)
#     return merged,writer





def final_run():
    session=tf.Session()
    batch_size=50
    zdim= 100
    alpha=0.0001

    x_placeholder=tf.placeholder(name='input_image',shape=[None,28,28,1],dtype=tf.float32)

    ############## LOSS FUNCTIONS ##############
    ##get the generated image output from the generator
    G_z=generator(batch_size,zdim)

    ##get the prediction probablities based on sigmoid
    ##discriminator output for Real image
    D_x=discriminator(x_placeholder)
    ##discriminator output for a fake image
    D_z=discriminator(G_z,reuse=True)

    ##now for getting the costs that we will eventually minimize

    ##for the generator loss we want the output to be one for a fake image
    g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_z,targets=tf.ones_like(D_z)))

    ##for the discrimniator loss for real images- these images should be labelled one
    d_loss_x=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_x,targets=tf.fill([batch_size, 1], 0.9)))
    ##for the discrimniator loss for fake images which should be labelled 0
    d_loss_z=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_z,targets=tf.zeros_like(D_z)))
    d_loss=d_loss_x+d_loss_z
    #############################################

    ##make a list of all trainable variables
    tvars=tf.trainable_variables()
    ##list of discriminator trainable variables
    d_vars=[]
    for var in tvars:
        if('d_' in var.name):
            d_vars.append(var)
    ##list of generator trainable variables
    g_vars=[]
    for var in tvars:
        if('g_' in var.name):
            g_vars.append(var)



    # now we will optimizer/train our weights
    # first we will optimize and train the discriminator network and then we will train teh generator network
    # to fool the already present discriminator network

    with tf.variable_scope(tf.get_variable_scope(), reuse=True) as scope:
        ##We will use the Adam optimizer because it is really a great optimizer for stochastic gradient descent
        d_fake_train=tf.train.AdamOptimizer(alpha).minimize(d_loss_z, var_list=d_vars)
        d_real_train=tf.train.AdamOptimizer(alpha).minimize(d_loss_x, var_list=d_vars)

        ##once the discriminator is trained we will then train the generator
        g_train=tf.train.AdamOptimizer(alpha).minimize(g_loss,var_list=g_vars)

    init=tf.global_variables_initializer()
    tf.summary.scalar('Generator Loss', g_loss)
    tf.summary.scalar('Discriminator Real loss', d_loss_x)
    tf.summary.scalar('Discriminator Fake loss', d_loss_z)

    ##sanity check for checking how the discriminator performs on the fake images
    d_eval_fake = tf.reduce_mean(discriminator(generator(batch_size, zdim)))
    tf.summary.scalar('Fake Evaluation', d_eval_fake)
    ##sanity check for checking how the discriminator performs on real images
    d_eval_real = tf.reduce_mean(discriminator(x_placeholder))

    ##get a list of all images and show it on tensorboard
    generated_images = generator(batch_size, zdim)
    tf.summary.image('Generated images', generated_images, 10)
    merged = tf.summary.merge_all()
    logdir = "C:/Users/ezio/Desktop/GAN/one/Tensorboard/"
    writer = tf.summary.FileWriter(logdir, session.graph)
    # merged,writer=write_summary(g_loss=g_loss,d_loss_real=d_loss_x,d_loss_fake=d_loss_z,batch_size=batch_size, z_dim=zdim ,x=x_placeholder,sess=session)

    # now while training we will be preveinting three basic conditions and training conditionally

    # first creating the saver
    saver=tf.train.Saver()
    session.run(init)

    gLoss=0
    dRealLoss,dFakeLoss=1,1

    for i in range(total_epoch):
        # print(i)
        ## get the real image batch
        real_image_batch=mnist.train.next_batch(batch_size)[0].reshape([batch_size,28,28,1])
        if(dFakeLoss>0.6):
            ##this is the case when it marks generated as real, so discriminator is doing a bad job
            _,dRealLoss,dFakeLoss,gLoss = session.run([d_fake_train,d_loss_x,d_loss_z,g_loss],feed_dict={x_placeholder:real_image_batch})

        if(gLoss>0.5):
            _,dRealLoss,dFakeLoss,gLoss= session.run([g_train,d_loss_x,d_loss_z,g_loss],feed_dict={x_placeholder:real_image_batch})

        if(dRealLoss>0.45):
            _,dRealLoss,dFakeLoss,gLoss=session.run([d_real_train,d_loss_x,d_loss_z,g_loss],feed_dict={x_placeholder:real_image_batch})

        ##writing the summary
        if(i%10==0):
            real_image_batch=mnist.validation.next_batch(batch_size)[0].reshape([batch_size,28,28,1])
            summary=session.run(merged,feed_dict={x_placeholder:real_image_batch})
            writer.add_summary(summary)
            print(i," >>  Losses:  Generator Loss: ",gLoss,"  Discriminator Real:  ",dRealLoss,"  Discriminator Fake: ",dFakeLoss)

        # if i % 10000 == 0:
        #     # Periodically display a sample image in the notebook
        #     # (These are also being sent to TensorBoard every 10 iterations)
        #     images = session.run(generator(1, zdim))
        #     d_result = session.run(discriminator(x_placeholder), {x_placeholder: images})
        #     print("TRAINING STEP", i, "AT", datetime.datetime.now())
        #     for j in range(1):
        #         # print("Discriminator classification", d_result[j])
        #         im = images[j, :, :, 0]
        #         # im.reshape([28, 28])
        #         # fl = "capturedframes/" + str(i) + ".jpg"
        #         plt.imshow(im.reshape([28, 28]), cmap='Greys')
        #         # cv2.imwrite(fl, im)
        #         plt.show()

        if(i%5000==0):
            save_path=saver.save(session,"model/test2/pretrained_gan.ckpt",global_step=i)
            print("SAVED TO",save_path)

final_run()







