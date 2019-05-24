"""
Our work is based upon and adpated from https://github.com/hwalsuklee/tensorflow-generative-model-collections/
"""

import cv2
import os
import numpy as np
from PIL import Image

def video_to_frames(filename, resize_h=128, resize_w=128):
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

def load_data(dataset_name, num_train, num_test, resize_h=128, resize_w=128):
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
        print(i)
        print(train_files[i])
        trX[i, ...] = video_to_frames(train_dir+train_files[i], resize_h, resize_w)
    teX = np.empty((num_test, num_frames, resize_h, resize_w, 3), dtype=np.float)
    for i in range(num_test):
        print(i)
        print(test_files[i])
        teX[i, ...] = video_to_frames(test_dir+test_files[i], resize_h, resize_w)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(trX)
    np.random.shuffle(teX)
    X = np.concatenate((trX, teX), axis=0)

    return X / 255.

#-*- coding: utf-8 -*-
from __future__ import division
import time
import tensorflow as tf

from ops import *
from utils import *

import prior_factory as prior

import imageio

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

class CVAE(object):
    model_name = "CVAE"     # name for checkpoint

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

        if dataset_name == 'skiing' or dataset_name == 'erupting':
            # parameters
            self.input_height = 128
            self.input_width = 128
            self.output_height = 128
            self.output_width = 128
            self.input_num_frames = 30
            self.output_num_frames = 30

            self.z_dim = z_dim         # dimension of noise-vector
            self.y_height = 128       # dimension of condition-vector (label)
            self.y_width  = 128
            self.c_dim = 3             # rgb color

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5
            self.lambda1 = 0.001 # hyperparameter for L2 loss between generated videos and given ones
            self.lambda2 = 0.0  # hyperparameter for L2 loss between generated initial/final frames and given ones

            # test
            self.sample_num = 16  # number of generated videos to be saved

            # load data
            self.data_X = load_data(self.dataset_name, self.num_train, self.num_test)

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size
        else:
            raise NotImplementedError

    def image_encoder(self, x_img, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        # Here we encode an image into a vector, NOT mean and std of a Gaussian distribution
        with tf.variable_scope("img_encoder", reuse=reuse):
            net = lrelu(conv2d(x_img, 16, 4, 4, 4, 4, name='img_en_conv1'))
            net = lrelu(bn(conv2d(net, 32, 4, 4, 2, 2, name='img_en_conv2'), is_training=is_training, scope='img_en_bn2'))
            net = tf.reshape(net, [self.batch_size, -1])
            net = lrelu(bn(linear(net, 64, scope='img_en_fc3'), is_training=is_training, scope='img_en_bn3'))
            enc = linear(net, self.img_enc_dim, scope='img_en_fc4')

        return enc
    
    # Gaussian encoder
    def video_encoder(self, x, is_training=True, reuse=False):
        with tf.variable_scope("encoder", reuse=reuse):
            # format: conv3d(input_, output_dim, k_t=3, k_h=5, k_w=5, d_t=3, d_h=2, d_w=2, stddev=0.02, name="conv3d")
            # output_dim means number of channels
            net = lrelu(conv3d(x, 32, 3, 4, 4, 3, 4, 4, name='en_conv1'))
            net = lrelu(bn(conv3d(net, 64, 3, 4, 4, 3, 4, 4, name='en_conv2'), is_training=is_training, scope='en_bn2'))
            net = tf.reshape(net, [self.batch_size, -1])
            net = lrelu(bn(linear(net, 128, scope='en_fc3'), is_training=is_training, scope='en_bn3'))
            gaussian_params = linear(net, 2 * self.z_dim, scope='en_fc4')

            # The mean parameter is unconstrained
            mean = gaussian_params[:, :self.z_dim]
            # The standard deviation must be positive. Parametrize with a softplus and
            # add a small epsilon for numerical stability
            stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, self.z_dim:])

        return mean, stddev

    # Bernoulli decoder
    def decoder(self, z, y1, y2, is_training=True, reuse=False):
        # Network Architecture is similar to that infoGAN (https://arxiv.org/abs/1606.03657)
        # Except we changed this to 3d deconvolution
        with tf.variable_scope("decoder", reuse=reuse):
            # merge noise and two encoded image vectors
            z = concat([z, y1, y2], 1)

            net = tf.nn.relu(bn(linear(z, 256, scope='de_fc1'), is_training=is_training, scope='de_bn1'))
            net = tf.nn.relu(bn(linear(net, 32 * 5 * 8 * 8, scope='de_fc2'), is_training=is_training, scope='de_bn2'))
            net = tf.reshape(net, [self.batch_size, 5, 8, 8, 32])
            # format: deconv3d(input_, output_shape, k_t=3, k_h=5, k_w=5, d_t=3, d_h=2, d_w=2, name="deconv3d", stddev=0.02, with_w=False)
            # formulas for output_size:
            # when padding is "same":  output_size = input_size * stride
            # when padding is "valid": output_size = (input_size-1)*stride+kernel_size
            # the default for tf.nn.conv3d_transpose is "same" padding
            net = tf.nn.relu(
                bn(deconv3d(net, [self.batch_size, 15, 32, 32, 8], 3, 4, 4, 3, 4, 4, name='de_dc3'), is_training=is_training,
                   scope='de_bn3'))

            out = tf.nn.sigmoid(deconv3d(net, [self.batch_size, 30, 128, 128, 3], 3, 4, 4, 2, 4, 4, name='de_dc4'))

            return out
        
    # extra processing after decoder, based on Geoff's proposal
    def process_post_decoder(self, x_out, img1, is_training=True, reuse=False):
        with tf.variable_scope("process_post_decoder", reuse=reuse):
            img1_broadcast = tf.transpose(tf.broadcast_to(img1, [30, self.batch_size, 128, 128, 3]), [1, 0, 2, 3, 4])
            # post = tf.multiply(img1_broadcast, x_out)
            post = -1.0 + 2.0*x_out + img1_broadcast # shape is [bs, 30, 128, 128, 3], same as self.inputs
            # post = tf.clip_by_value(post, 1e-8, 1 - 1e-8)
        return post


    def build_model(self):
        # some parameters
        video_dims = [30, self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size

        """ Graph Input """
        # videos
        self.inputs = tf.placeholder(tf.float32, [bs] + video_dims, name='real_videos')
        
        # initial and final frames from original videos
        self.img1 = tf.placeholder(tf.float32, [bs, 128, 128, 3], name='img1')
        self.img2 = tf.placeholder(tf.float32, [bs, 128, 128, 3], name='img2')

        # noises
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')
        
        """ Images To Vectors """
        # use self.image_encoder(x_img, is_training=True, reuse=False)
        y1 = self.image_encoder(self.img1, is_training=True, reuse=tf.AUTO_REUSE) # is this reuse correct??
        y2 = self.image_encoder(self.img2, is_training=True, reuse=tf.AUTO_REUSE)

        """ Loss Function """
        # encoding
        mu, sigma = self.video_encoder(self.inputs, is_training=True, reuse=False)

        # sampling by re-parameterization technique
        z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

        # decoding
        out1 = self.decoder(z, y1, y2, is_training=True, reuse=False)
        
        # post processing
        self.out = self.process_post_decoder(out1, self.img1, is_training=True, reuse=False)

        # loss
        # For mariginal log likelihood:
        # cross-entropy version: remember pixel values have been normalized to between 0 and 1
        # marginal_likelihood = self.lambda1*tf.reduce_sum(self.inputs * tf.log(self.out) + (1 - self.inputs) * tf.log(1 - self.out), [1, 2, 3, 4])
        # L2 version
        marginal_likelihood = -self.lambda1*tf.reduce_sum(tf.square(self.inputs - self.out), [1,2,3,4])
        marginal_likelihood = marginal_likelihood-self.lambda2*tf.reduce_sum(tf.square(self.inputs[:,0,...]-self.out[:,0,...]),[1,2,3])
        marginal_likelihood = marginal_likelihood-self.lambda2*tf.reduce_sum(tf.square(self.inputs[:,-1,...]-self.out[:,-1,...]),[1,2,3])

        KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, [1])

        self.neg_loglikelihood = -tf.reduce_mean(marginal_likelihood)
        self.KL_divergence = tf.reduce_mean(KL_divergence)

        ELBO = -self.neg_loglikelihood - self.KL_divergence

        self.loss = -ELBO

        """ Training """
        # optimizers
        t_vars = tf.trainable_variables()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optim = tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1) \
                      .minimize(self.loss, var_list=t_vars)

        """" Testing """
        # for test
        # remember y1, y2 depend on graph inputs self.img1 and self.img2
        # and remember the post processing step
        fake_out = self.decoder(self.z, y1, y2, is_training=False, reuse=True)
        self.fake_videos = self.process_post_decoder(fake_out, self.img1, is_training=False, reuse=True)

        """ Summary """
        nll_sum = tf.summary.scalar("nll", self.neg_loglikelihood)
        kl_sum = tf.summary.scalar("kl", self.KL_divergence)
        loss_sum = tf.summary.scalar("loss", self.loss)

        # final summary operations
        self.merged_summary_op = tf.summary.merge_all()

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = prior.gaussian(self.batch_size, self.z_dim)

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
                batch_videos = self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                
                # Extract initial and final frames
                batch_images1 = batch_videos[:,0,:,:,:].copy()
                batch_images2 = batch_videos[:,-1,:,:,:].copy()

                # update autoencoder
                _, summary_str, loss, nll_loss, kl_loss = self.sess.run([self.optim, self.merged_summary_op, self.loss, self.neg_loglikelihood, self.KL_divergence],
                                               feed_dict={self.inputs: batch_videos, self.z: batch_z, self.img1:batch_images1,
                                                         self.img2:batch_images2})
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f, nll: %.8f, kl: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, loss, nll_loss, kl_loss))

                # save training results for every 100 steps
                if np.mod(counter, 100) == 0:
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
                        
                        img1_array = batch_images1[ind_vid,...]*255.
                        img1_array = img1_array.astype(np.uint8)
                        img1 = Image.fromarray(img1_array, 'RGB')
                        img1.save(uri_im1)
                        
                        

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            # self.save(self.checkpoint_dir, counter)

            # show temporal results
            # self.visualize_results(epoch)

        # save model for final step
        # self.save(self.checkpoint_dir, counter)

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
            
            
# open session
tf.reset_default_graph()
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    # declare instance for CVAE
    arch = None # architecture
    # CVAE __init__(self, sess, epoch, batch_size, z_dim, img_enc_dim, dataset_name, checkpoint_dir, result_dir, log_dir, 
    # num_train, num_test):

    dataset_name='erupting'
    checkpoint_dir = './checkpoint/'
    result_dir = './result/'
    log_dir = './log/'
    arch = CVAE(sess,
                 epoch=200,
                 batch_size=10,
                 z_dim=200,
                 img_enc_dim=100,
                 dataset_name=dataset_name,
                 checkpoint_dir=checkpoint_dir,
                 result_dir=result_dir,
                 log_dir=log_dir,
                 num_train=100,
                 num_test=0)

    # build graph
    arch.build_model()

    # show network architecture
    show_all_variables()

    # launch the graph in a session
    arch.train()
    print(" [*] Training finished!")

    # visualize learned generator
    # arch.visualize_results(args.epoch-1)
    # print(" [*] Testing finished!")

