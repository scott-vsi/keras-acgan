#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
file: mnist_acgan.py
author: Luke de Oliveira (lukedeo@vaitech.io)

Train an Auxiliary Classifier Generative Adversarial Network (ACGAN) on the
MNIST dataset. See https://arxiv.org/abs/1610.09585 for more details.

You should start to see reasonable images after ~5 epochs, and good images
by ~15 epochs. You should use a GPU, as the convolution-heavy operations are
very slow on the CPU. Prefer the TensorFlow backend if you plan on iterating, as
the compilation time can be a blocker using Theano.

Timings:

Hardware           | Backend | Time / Epoch
-------------------------------------------
 CPU               | TF      | 3 hrs
 Titan X (maxwell) | TF      | 4 min
 Titan X (maxwell) | TH      | 7 min

Consult https://github.com/lukedeo/keras-acgan for more information and
example output
"""
from __future__ import print_function

from collections import defaultdict
import cPickle as pickle
from PIL import Image

from six.moves import range
import argparse

import keras.backend as K
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.layers.noise import GaussianNoise
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
import numpy as np

np.random.seed(1337)

K.set_image_dim_ordering('tf')


def build_generator(latent_size, is_pan=False, im_size=28):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 3, 56, 56)

    nb_channels = 1 if is_pan else 3
    upsampling_factor = 4 # 2**number of UpSamplings
    lowrez = im_size // upsampling_factor

    cnn = Sequential()

    cnn.add(Dense(1024, input_dim=latent_size, activation='relu'))
    cnn.add(Dense(128 * lowrez * lowrez, activation='relu'))
    cnn.add(Reshape((lowrez, lowrez, 128)))

    # upsample to (..., 14, 14)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(256, 5, 5, border_mode='same', init='glorot_normal'))
    cnn.add(BatchNormalization(axis=-1)) # set axis to normalize per feature map (channels axis)
    cnn.add(Activation('relu'))

    # upsample to (..., 28, 28)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(128, 5, 5, border_mode='same', init='glorot_normal'))
    cnn.add(BatchNormalization(axis=-1)) # set axis to normalize per feature map (channels axis)
    cnn.add(Activation('relu'))

    # upsample to (..., 56, 56)
    #cnn.add(UpSampling2D(size=(2, 2)))
    #cnn.add(Convolution2D(64, 5, 5, border_mode='same', init='glorot_normal'))
    #cnn.add(Activation('relu'))

    # take a channel axis reduction
    cnn.add(Convolution2D(nb_channels, 2, 2, border_mode='same', init='glorot_normal'))
    cnn.add(Activation('tanh'))

    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size, ))

    # this will be our label
    image_class = Input(shape=(1,), dtype='int32')

    # 10 classes in MNIST
    cls = Flatten()(Embedding(10, latent_size,
                              init='glorot_normal')(image_class))

    # hadamard product between z-space and a class conditional embedding
    h = merge([latent, cls], mode='mul')

    fake_image = cnn(h)

    return Model(input=[latent, image_class], output=fake_image)


def build_discriminator(is_pan=False, im_size=28, nb_kernels=32):
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper

    nb_channels = 1 if is_pan else 3

    cnn = Sequential()

    #cnn.add(GaussianNoise(0.12, input_shape=(im_size, im_size, nb_channels)))

    cnn.add(Convolution2D(nb_kernels*1, 3, 3, border_mode='same', subsample=(2, 2),
            input_shape=(im_size, im_size, nb_channels)))
    # the paper does not include BN here
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(nb_kernels*2, 3, 3, border_mode='same', subsample=(1, 1)))
    cnn.add(BatchNormalization(axis=-1)) # set axis to normalize per feature map (channels axis)
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(nb_kernels*4, 3, 3, border_mode='same', subsample=(2, 2)))
    cnn.add(BatchNormalization(axis=-1)) # set axis to normalize per feature map (channels axis)
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(nb_kernels*8, 3, 3, border_mode='same', subsample=(1, 1)))
    cnn.add(BatchNormalization(axis=-1)) # set axis to normalize per feature map (channels axis)
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    #cnn.add(Convolution2D(nb_kernels*16, 3, 3, border_mode='same', subsample=(2, 2)))
    #cnn.add(LeakyReLU())
    #cnn.add(Dropout(0.3))

    #cnn.add(Convolution2D(nb_kernels*32, 3, 3, border_mode='same', subsample=(1, 1)))
    #cnn.add(LeakyReLU())
    #cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    image = Input(shape=(im_size, im_size, nb_channels))

    features = cnn(image)

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    aux = Dense(10, activation='softmax', name='auxiliary')(features)

    return Model(input=image, output=[fake, aux])

def load_data(nb_images=None, nb_images_per_label=None, is_pan=False, im_size=56):
    # nb_images : number of images to load
    # nb_images_per_label : number of images per label to load
    # if nb_images is set and nb_images_per_label is None, images are drawn
    # from categories in proportion to their frequency in the dataset.
    # if nb_images_per_label is set and nb_images is None, the categories
    # are re-ballanced
    # is_pan : data conforms to the mnist data shape

    import os, os.path as path
    from glob import glob
    import scipy.misc as misc
    import itertools as it

    #import sklearn.preprocessing
    #files = glob('/like_mnist@2x/*/*.JPEG')
    #labels = sklearn.preprocessing.LabelEncoder().fit_transform(
    #    [path.split(path.split(f)[0])[1] for f in files])
    #images = [misc.imread(f) for f in files] # silently requires Pillow...

    filenames = []
    for root, dirs, files in os.walk('/data/by_yaw'):
        if dirs != []: continue # HACKY use walkdir
        files = [path.join(root, f) for f in files if '.JPEG' in f]
        files = np.random.permutation(files)
        filenames.append(files[:nb_images_per_label])

    labels = list(it.chain.from_iterable([[i]*len(lst) for i,lst in enumerate(filenames)]))
    filenames = list(it.chain.from_iterable(filenames))

    inds = np.random.permutation(len(filenames))[:nb_images]
    filenames, labels = [filenames[i] for i in inds], [labels[i] for i in inds]

    # silently requires Pillow...
    images = [misc.imread(f, mode='P' if is_pan else 'RGB') for f in filenames]
    images = [misc.imresize(im, size=(im_size,im_size), interp='bicubic') for im in images]

    # requires numpy > 1.10
    nb_train = int(0.9*len(filenames))
    X_train, y_train = np.stack(images[:nb_train]), labels[:nb_train]
    X_test, y_test = np.stack(images[nb_train:]), labels[nb_train:]

    def make_band_interleaved(pixel_interleaved_image):
        # nimages, nrows, ncols, nchannels
        return np.transpose(pixel_interleaved_image, (0,3,1,2))
    #if not is_pan: X_train, X_test = map(make_band_interleaved, [X_train, X_test])

    return (X_train, y_train), (X_test, y_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_epochs', type=int, help='number of training epochs, default=50', default=50)
    parser.add_argument('--batch_size', type=int, help='number of images per batch, default=100', default=100)
    parser.add_argument('--latent_size', type=int, help='size of the latent z vector, default=100', default=100)
    parser.add_argument('--adam_lr', type=float, help='learning rate (Adam), default=0.00005', default=0.00005)
    opt = parser.parse_args()

    # batch and latent size taken from the paper
    nb_epochs = opt.nb_epochs
    batch_size = opt.batch_size
    latent_size = opt.latent_size
    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    adam_lr = opt.adam_lr
    adam_beta_1 = 0.5
    is_pan = True

    # build the discriminator
    discriminator = build_discriminator(is_pan=is_pan, im_size=28)
    discriminator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )

    # build the generator
    generator = build_generator(latent_size, is_pan=is_pan, im_size=28)
    generator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
                      loss='binary_crossentropy')

    latent = Input(shape=(latent_size, ))
    image_class = Input(shape=(1,), dtype='int32')

    # get a fake image
    fake = generator([latent, image_class])

    # we only want to be able to train generation for the combined model
    discriminator.trainable = False
    fake, aux = discriminator(fake)
    combined = Model(input=[latent, image_class], output=[fake, aux])

    combined.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )

    # get our mnist data, and force it to be of shape (..., 3, 56, 56) with
    # range [-1, 1]
    #(X_train, y_train), (X_test, y_test) = load_data(is_pan=is_pan, nb_images_per_label=10000)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    if is_pan: X_train = np.expand_dims(X_train, axis=-1)

    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    if is_pan: X_test = np.expand_dims(X_test, axis=-1)

    nb_train, nb_test = X_train.shape[0], X_test.shape[0]

    train_history = defaultdict(list)
    test_history = defaultdict(list)

    for epoch in range(nb_epochs):
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        nb_batches = int(nb_train / batch_size)
        progress_bar = Progbar(target=nb_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []

        # TODO add shuffling

        for index in range(nb_batches):
            progress_bar.update(index)
            # generate a new batch of noise
            noise = np.random.uniform(-1, 1, (batch_size, latent_size))

            # get a batch of real images
            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            label_batch = y_train[index * batch_size:(index + 1) * batch_size]

            # sample some labels from p_c
            sampled_labels = np.random.randint(0, 10, batch_size)

            # generate a batch of fake images, using the generated labels as a
            # conditioner. We reshape the sampled labels to be
            # (batch_size, 1) so that we can feed them into the embedding
            # layer as a length one sequence
            generated_images = generator.predict(
                [noise, sampled_labels.reshape((-1, 1))], verbose=0)

            X = np.concatenate((image_batch, generated_images))
            y = np.array([1] * batch_size + [0] * batch_size)
            aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

            # see if the discriminator can figure itself out...
            epoch_disc_loss.append(discriminator.train_on_batch(X, [y, aux_y]))

            # make new noise. we generate 2 * batch size here such that we have
            # the generator optimize over an identical number of images as the
            # discriminator
            noise = np.random.uniform(-1, 1, (2 * batch_size, latent_size))
            sampled_labels = np.random.randint(0, 10, 2 * batch_size)

            # we want to train the genrator to trick the discriminator
            # For the generator, we want all the {fake, not-fake} labels to say
            # not-fake
            trick = np.ones(2 * batch_size)

            epoch_gen_loss.append(combined.train_on_batch(
                [noise, sampled_labels.reshape((-1, 1))], [trick, sampled_labels]))

        print('\nTesting for epoch {}:'.format(epoch + 1))

        # evaluate the testing loss here

        # generate a new batch of noise
        noise = np.random.uniform(-1, 1, (nb_test, latent_size))

        # sample some labels from p_c and generate images from them
        sampled_labels = np.random.randint(0, 10, nb_test)
        generated_images = generator.predict(
            [noise, sampled_labels.reshape((-1, 1))], verbose=False)

        X = np.concatenate((X_test, generated_images))
        y = np.array([1] * nb_test + [0] * nb_test)
        aux_y = np.concatenate((y_test, sampled_labels), axis=0)

        # see if the discriminator can figure itself out...
        discriminator_test_loss = discriminator.evaluate(
            X, [y, aux_y], verbose=False)

        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        # make new noise
        noise = np.random.uniform(-1, 1, (2 * nb_test, latent_size))
        sampled_labels = np.random.randint(0, 10, 2 * nb_test)

        trick = np.ones(2 * nb_test)

        generator_test_loss = combined.evaluate(
            [noise, sampled_labels.reshape((-1, 1))],
            [trick, sampled_labels], verbose=False)

        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        # generate an epoch report on performance
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)

        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
            'component', *discriminator.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
        print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)',
                             *test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)',
                             *test_history['discriminator'][-1]))

        # save weights every epoch
        generator.save_weights(
            'params_generator_epoch_{0:03d}.hdf5'.format(epoch), True)
        discriminator.save_weights(
            'params_discriminator_epoch_{0:03d}.hdf5'.format(epoch), True)

        # generate some digits to display
        noise = np.random.uniform(-1, 1, (100, latent_size))

        sampled_labels = np.array([
            [i] * 10 for i in range(10)
        ]).reshape(-1, 1)

        # get a batch to display
        generated_images = generator.predict(
            [noise, sampled_labels], verbose=0)

        def make_pixel_interleaved(band_interleaved_image):
            # nimages, nrows, ncols, nchannels
            return np.transpose(band_interleaved_image, (0,2,3,1))
        def make_grid(tensor, ncols=10):
            nb_images = tensor.shape[0]
            tensor = np.pad(tensor, pad_width=[(0,np.mod(nb_images, ncols))]+[(0,0)]*3,
                    mode='constant', constant_values=0)
            def make_col(images):
                nb_images = images.shape[0]
                return np.squeeze(np.hstack(np.split(images, nb_images, axis=0)), axis=0)
            # REVIEW just do with a reshape
            return np.squeeze(np.hstack([make_col(r) for r in np.split(tensor, ncols)]))

        # arrange them into a grid
        #if not is_pan: im_grid = make_pixel_interleaved(im_grid)
        im_grid = make_grid((generated_images * 127.5 + 127.5).astype(np.uint8))

        Image.fromarray(im_grid).save(
            'plot_epoch_{0:03d}_generated.png'.format(epoch))

    pickle.dump({'train': train_history, 'test': test_history},
                open('acgan-history.pkl', 'wb'))
