"""
This file is based on and adapted from https://github.com/hwalsuklee/tensorflow-generative-model-collections/
"""

import cv2
import os
import numpy as np
from PIL import Image
import imageio

from __future__ import division
import time
import tensorflow as tf

from ops import *
from utils import *
import prior_factory as prior


def video_to_frames(filename, resize_h=64, resize_w=64):
    """                                                                                                           
    Extract every frame from a video, and keep one in every 3 frames, and resize each frame.
    For original video: expect 90 frames per 3 second video, each frame 256-by-256.
    If the video has less than 90 frames, fill all remaining frames up to num_frames
    """
    # an example for filename: 'videos/skiing_train/flickr-0-0-0-1-1-5-9-7-2400011597_4.mp4'
    vidcap = cv2.VideoCapture(filename)
    frameCount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #expect 30 frames to be extracted; if original video has less than
    #90 frames, fill the trailing ones with the last sampled frame
    reduced_frameCount = 30 

    data = np.empty((reduced_frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    
    fc = 0
    success = True
    while (fc < min(frameCount, 90) and success):
        if (fc%3==0):
            success, data[fc//3, ...] = vidcap.read()
        else:
            vidcap.read()
        fc += 1
    if frameCount < 90-3:
        for i in range(frameCount//3+1, 30):
            data[i, ...] = data[frameCount//3, ...]
        
    #print('Finished reading video')
        
    # keep only one frame out of every 3 frames, and resize every frame
    data_resized = np.empty((reduced_frameCount, resize_h, resize_w, 3), np.float)
    for i in range(reduced_frameCount):
        img = Image.fromarray(data[i,...], 'RGB')
        img = img.resize((resize_h, resize_w), Image.ANTIALIAS)
        data_resized[i, ...] = np.asarray(img, dtype=np.float)
        # below line just for testing purpose, make sure to comment out after testing!
        ###img.save('extracted_images/skiing_test_1_frame%d.jpg' % i)

    return data_resized

def load_data(dataset_name, num_train, num_test, test_files_list, resize_h=64, resize_w=64):
    data_dir = 'videos/'+dataset_name+'_' #dataset should be "skiing" or "erupting"
    #data_dir = os.path.join("videos/", dataset_name)

    train_dir = data_dir+'train/'
    test_dir  = data_dir+'val/'
    train_files = os.listdir(train_dir)
    test_files  = os.listdir(test_dir)
    if num_train > len(train_files):
        print('num_train is more than number of training examples, set equal.')
        num_train = len(train_files)
        print(num_train)
    if num_test > len(test_files):
        print('num_test is more than number of testing examples, set equal.')
        num_test = len(test_files)
        print(num_test)
    train_files = train_files[:num_train]
    test_files  = test_files[:num_test]
    
    num_frames = 30
    trX = np.empty((num_train, num_frames, resize_h, resize_w, 3), dtype=np.float)
    for i in range(num_train):
        #print(i)
        #print(train_files[i])
        trX[i, ...] = video_to_frames(train_dir+train_files[i], resize_h, resize_w)
    teX = np.empty((num_test, num_frames, resize_h, resize_w, 3), dtype=np.float)
    for i in range(num_test):
        #print(i)
        #print(test_files[i])
        test_files_list.append(test_dir+test_files[i])
        teX[i, ...] = video_to_frames(test_dir+test_files[i], resize_h, resize_w)

    seed = 547
    np.random.seed(seed)
    #np.random.shuffle(trX)
    #np.random.shuffle(teX)
    #X = np.concatenate((trX, teX), axis=0)

    return trX / 255., teX / 255.

def conv3d(input_, output_dim, k_t=3, k_h=5, k_w=5, d_t=3, d_h=2, d_w=2, stddev=0.02, name="conv3d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_t, k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv3d(input_, w, strides=[1, d_t, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def deconv3d(input_, output_shape, k_t=3, k_h=5, k_w=5, d_t=3, d_h=2, d_w=2, name="deconv3d", stddev=0.02, with_w=False):
    with tf.variable_scope(name):
        # filter : [depth, height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_t, k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv3d_transpose(input_, w, output_shape=output_shape, strides=[1, d_t, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv
        
def conv_cond_concat_3d(x, y):
    """Video version: Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], x_shapes[3], y_shapes[4]])], 4)

class GAN(object):
    model_name = "GAN"     # name for checkpoint

    def __init__(self, sess, epoch, batch_size, z_dim, img_enc_dim, dataset_name, checkpoint_dir, result_dir, log_dir, 
                 num_train, num_test):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.num_train = num_train
        self.num_test = num_test
        self.img_enc_dim = img_enc_dim
        self.test_files_list = []
        self.generated_files_list = []

        if dataset_name == 'skiing' or dataset_name == 'erupting':
            # parameters
            self.input_height = 64
            self.input_width = 64
            self.output_height = 64
            self.output_width = 64
            self.input_num_frames = 30
            self.output_num_frames = 30

            self.z_dim = z_dim         # dimension of noise-vector
            self.y_height = 64       # dimension of condition-vector (label)
            self.y_width  = 64
            self.c_dim = 3             # rgb color

            # train
            self.d_update_times = 4 # number of times D is updated per iteration
            self.g_update_times = 1 # number of times G is updated per iteration
            self.d_learning_rate = 0.0002 # learning rate for D
            self.g_learning_rate = 0.0002 # learing rate for G, 0.0002 is the default from original github repository
            self.beta1 = 0.5
            self.lambda1 = 0.0 # not used: hyperparameter for L2 loss between generated videos and given ones
            self.lambda2 = 0.02  # hyperparameter for L1 or L2 loss between generated initial/final frames and given ones

            # test
            self.sample_num = 16  # number of generated videos to be saved

            # load data
            self.data_X_train, self.data_X_test = load_data(self.dataset_name, self.num_train, self.num_test, self.test_files_list)

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X_train) // self.batch_size
        else:
            raise NotImplementedError
            
            
    def image_encoder(self, x_img, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        # Here we encode an image into a vector, NOT mean and std of a Gaussian distribution
        
        # This image_encoder will ONLY be used for generator, NOT DISCRIMINATOR
        
        with tf.variable_scope("img_encoder", reuse=reuse):
            net = lrelu(conv2d(x_img, 16, 4, 4, 2, 2, name='g_img_en_conv1'))
            net = lrelu(bn(conv2d(net, 32, 3, 3, 1, 1, name='g_img_en_conv2'), is_training=is_training, scope='g_img_en_bn2'))
            net = lrelu(bn(conv2d(net, 64, 4, 4, 2, 2, name='g_img_en_conv3'), is_training=is_training, scope='g_img_en_bn3'))
            net = lrelu(bn(conv2d(net, 128, 3, 3, 1, 1, name='g_img_en_conv4'), is_training=is_training, scope='g_img_en_bn4'))
            net = tf.reshape(net, [self.batch_size, -1])
            net = lrelu(bn(linear(net, 256, scope='g_img_en_fc6'), is_training=is_training, scope='g_img_en_bn6'))
            enc = linear(net, self.img_enc_dim, scope='g_img_en_fc7')

        return enc

            
    def discriminator(self, x, is_training=True, reuse=False):
        # Network Architecture is similar to that infoGAN (https://arxiv.org/abs/1606.03657)
        # Except we changed this to 3d deconvolution
        # For GAN (instead of CGAN), we will not feed the initial and final frames into the discriminator
        with tf.variable_scope("discriminator", reuse=reuse):
            
            # format: conv3d(input_, output_dim, k_t=3, k_h=5, k_w=5, d_t=3, d_h=2, d_w=2, stddev=0.02, name="conv3d")
            # output_dim means number of channels
            net = lrelu(conv3d(x, 16, 3, 4, 4, 1, 2, 2, name='d_conv1'))
            net = lrelu(bn(conv3d(net, 32, 3, 4, 4, 1, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
            net = lrelu(bn(conv3d(net, 64, 3, 4, 4, 1, 2, 2, name='d_conv3'), is_training=is_training, scope='d_bn3'))
            #net = lrelu(bn(conv3d(net, 256, 3, 4, 4, 1, 2, 2, name='d_conv4'), is_training=is_training, scope='d_bn4'))
            net = tf.reshape(net, [self.batch_size, -1])
            net = lrelu(bn(linear(net, 128, scope='d_fc5'), is_training=is_training, scope='d_bn5'))
            out_logit = linear(net, 1, scope='d_fc6')
            out = tf.nn.sigmoid(out_logit)

            return out, out_logit, net

    def generator(self, z, y1, y2, is_training=True, reuse=False):
        # same as the decoder for our CVAE
        with tf.variable_scope("generator", reuse=reuse):
            
            # merge noise and two encoded image vectors
            z = concat([z, y1, y2], 1)

            #net = tf.nn.relu(bn(linear(z, 512, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
            net = tf.nn.relu(bn(linear(z, 256 * 5 * 8 * 8, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
            net = tf.reshape(net, [self.batch_size, 5, 8, 8, 256])
            # format: deconv3d(input_, output_shape, k_t=3, k_h=5, k_w=5, d_t=3, d_h=2, d_w=2, name="deconv3d", stddev=0.02, with_w=False)
            # formulas for output_size:
            # when padding is "same":  output_size = input_size * stride
            # when padding is "valid": output_size = (input_size-1)*stride+kernel_size
            # the default for tf.nn.conv3d_transpose is "same" padding
            net = tf.nn.leaky_relu(
                bn(deconv3d(net, [self.batch_size, 5, 8, 8, 128], 1, 2, 2, 1, 1, 1, name='g_dc3'), is_training=is_training,
                   scope='g_bn3'))
            
            net = tf.nn.leaky_relu(
                bn(deconv3d(net, [self.batch_size, 15, 16, 16, 64], 3, 2, 2, 3, 2, 2, name='g_dc4'), is_training=is_training,
                   scope='g_bn4'))
            
            net = tf.nn.leaky_relu(
                bn(deconv3d(net, [self.batch_size, 15, 32, 32, 32], 3, 2, 2, 1, 2, 2, name='g_dc5'), is_training=is_training,
                   scope='g_bn5'))
            
            net = tf.nn.leaky_relu(
                bn(deconv3d(net, [self.batch_size, 15, 64, 64, 16], 3, 4, 4, 1, 2, 2, name='g_dc6'), is_training=is_training,
                   scope='g_bn6'))

            out = tf.nn.sigmoid(deconv3d(net, [self.batch_size, 30, 64, 64, 3], 2, 4, 4, 2, 1, 1, name='g_dc7'))

            return out

    def build_model(self):
        # some parameters
        video_dims = [30, self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size
        
        """ Graph Input """
        # videos
        self.inputs = tf.placeholder(tf.float32, [bs] + video_dims, name='real_videos')
        
        # initial and final frames from original videos
        self.img1 = tf.placeholder(tf.float32, [bs, 1, 64, 64, 3], name='img1')
        self.img2 = tf.placeholder(tf.float32, [bs, 1, 64, 64, 3], name='img2')

        # noises
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')
        self.z_test = tf.placeholder(tf.float32, [self.num_test, self.z_dim], name='z_test')
        
        """ Images To Vectors """
        # use self.image_encoder(x_img, is_training=True, reuse=False)
        y1 = self.image_encoder(self.img1[:,0,:,:,:], is_training=True, reuse=tf.AUTO_REUSE) # is this reuse correct??
        y2 = self.image_encoder(self.img2[:,0,:,:,:], is_training=True, reuse=tf.AUTO_REUSE)


        """ Loss Function """
        # x = logits, z = labels. The logistic loss is z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        
        # output of D for real videos
        D_real, D_real_logits, _ = self.discriminator(self.inputs, is_training=True, reuse=False)

        # output of D for fake videos
        G = self.generator(self.z, y1, y2, is_training=True, reuse=False)
        D_fake, D_fake_logits, _ = self.discriminator(G, is_training=True, reuse=True)

        # get loss for discriminator
        d_loss_real = tf.reduce_mean(
            #tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
            # one-sided label smoothing (for positive/real videos only)
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.fill(tf.shape(D_real), 0.9)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))

        self.d_loss = d_loss_real + d_loss_fake

        # get loss for generator
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))
        
        # add L2 distance loss with given initial and final frames to generator loss
        #loss_init = self.lambda2*tf.reduce_mean(tf.reduce_sum(tf.square(self.img1[:,0,:,:,:]-G[:,0,...]),[1,2,3]))
        #loss_fin  = self.lambda2*tf.reduce_mean(tf.reduce_sum(tf.square(self.img2[:,0,:,:,:]-G[:,-1,...]),[1,2,3]))
        
        # or L1 version
        loss_init = self.lambda2*tf.reduce_mean(tf.reduce_sum(tf.abs(self.img1[:,0,:,:,:]-G[:,0,...]),[1,2,3]))
        loss_fin  = self.lambda2*tf.reduce_mean(tf.reduce_sum(tf.abs(self.img2[:,0,:,:,:]-G[:,-1,...]),[1,2,3]))
        self.dist_loss = loss_init + loss_fin


        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            # Remember: k update for D / 1 update for G
            self.d_optim = tf.train.AdamOptimizer(self.d_learning_rate, beta1=self.beta1) \
                      .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.g_learning_rate, beta1=self.beta1) \
                      .minimize(self.g_loss + self.dist_loss, var_list=g_vars)
            

        """" Testing """
        # for test
        self.fake_videos = self.generator(self.z, y1, y2, is_training=False, reuse=True)
        
        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        dist_loss_sum = tf.summary.scalar("dist_loss", self.dist_loss)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])
        self.dist_sum = dist_loss_sum # is this correct?

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = np.random.randn(self.batch_size, self.z_dim).astype(np.float32)
        # instead of np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim))

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_videos = self.data_X_train[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_z = np.random.randn(self.batch_size, self.z_dim).astype(np.float32)
                #batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                
                # Extract initial and final frames
                num_frames = batch_videos.shape[1]
                batch_images1 = batch_videos[:,0:1,:,:,:]
                batch_images2 = batch_videos[:,num_frames-1:num_frames,:,:,:]

                # update D network
                for i in range(self.d_update_times):
                    _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                                           feed_dict={self.inputs: batch_videos, 
                                                                      self.img1: batch_images1, 
                                                                      self.img2: batch_images2,self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # update G network
                for i in range(self.g_update_times):
                    _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss],
                                                           feed_dict={self.img1: batch_images1, self.img2: batch_images2,
                                                                      self.z: batch_z})
                self.writer.add_summary(summary_str, counter)
                
                # write out dist_loss
                summary_str, dist_loss = self.sess.run([self.dist_sum, self.dist_loss],
                                                       feed_dict={self.img1: batch_images1, self.img2: batch_images2,
                                                                  self.z: batch_z})
                self.writer.add_summary(summary_str, counter)
                

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, dist_loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss, dist_loss))
                
                # save training results for every 100 steps
                if np.mod(counter, 90) == 0:
                    samples = self.sess.run(self.fake_videos,
                                            feed_dict={self.z: self.sample_z,self.img1:batch_images1,self.img2:batch_images2})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    samples = np.clip(samples, 0.0, 1.0)
                    samples = samples*255.
                    samples = samples.astype(np.uint8)
                    
                    for ind_vid in range(tot_num_samples):
                        uri = './' + check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name 
                        uri_vid = uri + '_train_{:02d}_{:04d}_{:04d}.mp4'.format(epoch, idx, ind_vid)
                        uri_im1 = uri + '_train_{:02d}_{:04d}_{:04d}_img1.jpg'.format(epoch, idx, ind_vid)
                        
                        imageio.mimwrite(uri_vid, samples[ind_vid], fps=10)
                        
                        img1_array = batch_images1[ind_vid,0,...]*255.
                        img1_array = img1_array.astype(np.uint8)
                        img1 = Image.fromarray(img1_array, 'RGB')
                        img1.save(uri_im1)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            if (epoch%100 == 0 and epoch > 0):
                self.save(self.checkpoint_dir, counter)

            # show temporal results
            # self.visualize_results(epoch)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
            
            
# Main part: run training and testing
# open session
tf.reset_default_graph()
with tf.device('/gpu:0'):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # declare instance for GAN
        arch = None # architecture
        # GAN __init__(self, sess, epoch, batch_size, z_dim, img_enc_dim, dataset_name, checkpoint_dir, result_dir, log_dir, 
        # num_train, num_test):

        dataset_name='erupting'
        checkpoint_dir = './checkpoint/'
        result_dir = './result/GAN/'
        log_dir = './log/'
        arch = GAN(sess,
                     epoch=1800,
                     batch_size=100,
                     z_dim=600,
                     img_enc_dim=1000,
                     dataset_name=dataset_name,
                     checkpoint_dir=checkpoint_dir,
                     result_dir=result_dir,
                     log_dir=log_dir,
                     num_train=2000,
                     num_test=100)


        # build graph
        arch.build_model()

        # show network architecture
        show_all_variables()

        # launch the graph in a session
        arch.train()
        print(" [*] Training finished!")


