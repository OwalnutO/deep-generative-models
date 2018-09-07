"""
run_InfoGAN.py

Reference:
X. Chen, Y. Duan, R. Houthooft, J. Schulman, I. Sutskever, and P. Abbeel. Infogan:  interpretable representation learning by information maximizing generative adversarial nets. In NIPS, 2016.

LZhu, 2018-May
"""

# import general packages
import sys
import numpy as np
import scipy.io as sio
import os
os.environ['KERAS_BACKEND'] = 'tensorflow' # set up tensorflow backend for keras
from os import path
import matplotlib
if os.name != 'nt' and os.environ.get('DISPLAY','') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import keras packages
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Reshape, Flatten, Activation, Lambda, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras.utils import generic_utils
from keras.utils import plot_model  # for model visualization, need to install Graphviz (.msi for Windows), pydot (pip install), graphviz (pip install) and set PATH for Graphviz
from keras.optimizers import *
from keras import backend as K
K.set_image_data_format('channels_last')
from sklearn.utils.linear_assignment_ import linear_assignment



# remove duplicate data
def remove_duplicates(dep, data):
    _, idxUnq = np.unique(dep, return_index=True)  # sorted array
    dep = dep[idxUnq]
    if isinstance(data, list):
        data = [d for d in data]
    else:
        data = [data]
    for i in range(len(data)):
        data[i] = data[i][idxUnq]
    return dep, data


# set network weights trainability
def set_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


# normalize data to [a, b]
def normalize_data(data, a=0, b=1, method='sample'):
    if method is 'sample':  # normalize for each sample
        min_data = data.min(axis=tuple(np.arange(1, data.ndim)), keepdims=True)
        max_data = data.max(axis=tuple(np.arange(1, data.ndim)), keepdims=True)
    elif method is 'all':  # normalize across all samples
        min_data = data.min()
        max_data = data.max()
    else:
        raise NotImplementedError
    data_norm = ((b - a) * (data - min_data) / (max_data - min_data)) + a
    return data_norm, min_data, max_data


# denormalize data
def denormalize_data(data, min_data, max_data, a=0, b=1):
    data_denorm = ((data - a) * (max_data - min_data) / (b - a)) + min_data
    return data_denorm


def cluster_acc(y_true, y_pred):
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, w


# plot loss function
def plot_loss(losses, save=False, save_filename=None):
    plt.figure(figsize=(10, 10))
    g_loss = np.array(losses['g'])
    d_loss = np.array(losses['d'])
    plt.plot(g_loss[:, 1], label='G loss')
    plt.plot(g_loss[:, 2], label='Q loss')
    plt.plot(d_loss[:, 0], label='D loss')
    plt.plot(d_loss[:, 1], label='D mse')
    plt.legend()
    if save:
        if save_filename is not None:
            plt.savefig(save_filename)
    else:
        plt.show()
    plt.clf()
    plt.close('all')


# generate random noise and categorical latent variables
def get_noise(dim_noise, dim_cat, batch_size=32):
    noise = np.random.uniform(-1, 1, size=(batch_size, dim_noise))
    label = np.random.randint(0, dim_cat, size=(batch_size, 1))
    label = np_utils.to_categorical(label, num_classes=dim_cat)
    return noise, label


# plot generated images
def plot_gen(generator, dim, figsize=(10, 10), channel=0, save=False, save_filename=None, **kwargs):
    dim_noise = generator.layers[0].input_shape[1]
    dim_cat = generator.layers[1].input_shape[1]
    n_image_col = dim[1]
    plt.figure(figsize=figsize)
    for i in range(dim_cat):
        noise, _ = get_noise(dim_noise, dim_cat, batch_size=n_image_col)
        label = np.repeat(i, n_image_col).reshape(-1, 1)
        label = np_utils.to_categorical(label, num_classes=dim_cat)
        image_gen = generator.predict([noise, label])
        for j in range(n_image_col):
            plt.subplot(dim_cat, n_image_col, i*n_image_col+j+1)
            if isinstance(channel, int):
                plt.imshow(image_gen[j, :, :, channel], **kwargs)
            elif channel is 'color':
                plt.imshow(image_gen[j], **kwargs)
            plt.axis('off')
    if save:
        if save_filename is not None:
            plt.savefig(save_filename)
    else:
        plt.tight_layout()
        plt.show()
    plt.clf()
    plt.close('all')


def build_generator(n_class, n_rows, n_cols, n_out_ch=1, n_first_conv_ch=256, dim_noise=100):
    g_in_noise = Input(shape=(dim_noise,))
    g_in_cat = Input(shape=(n_class,))
    g_in = concatenate([g_in_noise, g_in_cat])
    x = Dense((n_rows//4) * (n_cols//4) * n_first_conv_ch)(g_in)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = Reshape((n_rows // 4, n_cols // 4, n_first_conv_ch))(x)
    x = Conv2DTranspose(n_first_conv_ch//2, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(n_first_conv_ch//4, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    g_out = Conv2DTranspose(n_out_ch, (4, 4), strides=1, padding='same', activation='sigmoid')(x)  # output in [0, 1]
    generator = Model([g_in_noise, g_in_cat], g_out, name='generator')
    print('Summary of Generator (for InfoGAN)')
    generator.summary()
    return generator


def build_disc_aux(n_class, n_rows, n_cols, n_in_ch=1, n_last_conv_ch=256, leaky_relu_alpha=0.2, lr=1e-4, beta_1=0.5, beta_2=0.9):
    d_opt = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2)
    d_in = Input(shape=(n_rows, n_cols, n_in_ch))
    x = Conv2D(n_last_conv_ch//4, (4, 4), strides=1, padding='same')(d_in)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = Conv2D(n_last_conv_ch//2, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    x = Conv2D(n_last_conv_ch, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=leaky_relu_alpha, name='d_feature')(x)
    d_out_base = Flatten()(x)

    # real probability output
    d_out_sigmoid = Dense(1, activation='sigmoid')(d_out_base)

    # categorical output
    x = Dense(1024)(d_out_base)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=leaky_relu_alpha)(x)
    c_out_softmax = Dense(n_class, activation='softmax')(x)

    discriminator = Model(d_in, d_out_sigmoid, name='discriminator')
    discriminator.compile(optimizer=d_opt, loss='binary_crossentropy', metrics=['mean_squared_error'])
    print('Summary of Discriminator (for InfoGAN)')
    discriminator.summary()

    classifier = Model(d_in, c_out_softmax, name='classifier')
    classifier.compile(optimizer=d_opt, loss='categorical_crossentropy', metrics=['mean_squared_error'])
    print('Summary of Classifier (for InfoGAN)')
    classifier.summary()

    return discriminator, classifier


def build_infogan(generator, discriminator, classifier, lr=1e-4, beta_1=0.5, beta_2=0.9):
    gan_opt = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2)
    gan_in_noise = Input(shape=generator.layers[0].input_shape[1:])
    gan_in_cat = Input(shape=generator.layers[1].input_shape[1:])
    gen_out = generator([gan_in_noise, gan_in_cat])
    set_trainable(discriminator, False)
    gan_out_sigmoid = discriminator(gen_out)
    gan_out_softmax = classifier(gen_out)
    gan = Model([gan_in_noise, gan_in_cat], [gan_out_sigmoid, gan_out_softmax])
    gan.compile(optimizer=gan_opt, loss=['binary_crossentropy', 'categorical_crossentropy'], loss_weights=[1, 1])
    print('Summary of the InfoGAN model')
    gan.summary()
    return gan


def train_infogan(image_set, generator, discriminator, gan, losses,
                  batch_size=32,
                  n_epochs=100,
                  save_every=10,
                  save_mode='multi_channel',
                  save_filename_prefix=None):
    label_smooth = 0.1  # label smoothing factor
    n_train = image_set.shape[0]
    dim_noise = generator.layers[0].input_shape[1]
    dim_cat = generator.layers[1].input_shape[1]
    n_ch = discriminator.layers[0].input_shape[-1]
    for ie in range(n_epochs):
        print('epoch: %d' % (ie + 1))
        idx_randperm = np.random.permutation(n_train)
        n_batches = n_train // batch_size
        progbar = generic_utils.Progbar(n_batches*batch_size)
        for ib in range(n_batches):
            # train discriminator
            set_trainable(discriminator, True)
            # train real batch
            idx_batch = idx_randperm[range(ib*batch_size, ib*batch_size+batch_size)]
            X_real = image_set[idx_batch]
            y_real = np.random.uniform(low=1-label_smooth, high=1, size=(batch_size,))  # label smoothing
            d_loss_real = discriminator.train_on_batch(X_real, y_real)
            # train fake batch
            noise_gen, label_gen = get_noise(dim_noise, dim_cat, batch_size=batch_size)  # generate noise and labels
            X_fake = generator.predict([noise_gen, label_gen])
            y_fake = np.random.uniform(low=0, high=label_smooth, size=(batch_size,))  # label smoothing
            d_loss_fake = discriminator.train_on_batch(X_fake, y_fake)
            # discriminator loss
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            losses['d'].append(d_loss)

            # train generator and classifier
            noise_train, label_train = get_noise(dim_noise, dim_cat, batch_size=batch_size)
            y_one = np.ones((batch_size,), dtype=np.float32)
            set_trainable(discriminator, False)
            # generator loss and classifier loss
            g_loss = gan.train_on_batch([noise_train, label_train], [y_one, label_train])
            losses['g'].append(g_loss)

            # update progress bar
            progbar.add(batch_size, values=[("G loss", g_loss[1]), ("Q loss", g_loss[2]),
                                            ("D loss", d_loss[0]), ("D mse", d_loss[1])])

        # plot interim results
        if ((ie + 1) % save_every == 0) or (ie == n_epochs - 1):
            if save_mode is 'multi_channel':
                # display generated images channel by channel
                for ic in range(n_ch):
                    save_filename_image_gen = '%s_cat_gen_ch%d_epoch%d.pdf' % (save_filename_prefix, ic, ie + 1)
                    plot_gen(generator, (dim_cat, 10), (15, 15), ic, True, save_filename_image_gen, cmap='gray')
            elif save_mode is 'color':
                save_filename_image_gen = '%s_cat_gen_epoch%d.pdf' % (save_filename_prefix, ie + 1)
                plot_gen(generator, (dim_cat, 10), (15, 15), 'color', True, save_filename_image_gen)

    # plot loss
    save_filename_loss = '%s_loss_epoch%d.pdf' % (save_filename_prefix, n_epochs)
    plot_loss(losses, True, save_filename_loss)



if __name__ == '__main__':

    losses = {'g': [], 'd': []}

    # sanity check on mnist
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_train_min, x_train_max = normalize_data(x_train, a=0, b=1, method='sample')
    x_test, x_test_min, x_test_max = normalize_data(x_test, a=0, b=1, method='sample')
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    n_row, n_col, n_ch = x_train.shape[1:]

    # training hyperparameters
    batch_size = 128
    n_epochs = 100
    n_save_every = 10
    g_lr = 2e-4
    g_beta1 = 0.5
    g_beta2 = 0.9
    d_lr = 2e-4
    d_beta1 = 0.5
    d_beta2 = 0.9
    dim_noise = 50
    n_class = 10

    # output folder
    output_path = './output/'
    if not path.exists(output_path):
        os.makedirs(output_path)
    save_mode = 'multi_channel'
    save_filename_prefix = '%sMNIST_infogan_noise%d_cat%d' % (output_path, dim_noise, n_class)

    # generator
    generator = build_generator(n_class, n_row, n_col, n_out_ch=n_ch, n_first_conv_ch=256, dim_noise=dim_noise)
    plot_model(generator, to_file='%s_generator_model.pdf' % save_filename_prefix, show_shapes=True)
    # discriminator and classifier
    discriminator, classifier = build_disc_aux(n_class, n_row, n_col, n_in_ch=n_ch, n_last_conv_ch=256, lr=d_lr)
    plot_model(discriminator, to_file='%s_discriminator_model.pdf' % save_filename_prefix, show_shapes=True)
    plot_model(classifier, to_file='%s_classifier_model.pdf' % save_filename_prefix, show_shapes=True)
    # combined InfoGAN network
    infogan = build_infogan(generator, discriminator, classifier, lr=g_lr)
    plot_model(infogan, to_file='%s_infogan_model.pdf' % save_filename_prefix, show_shapes=True)
    gan_weights_file = '%s_weights_epoch%d.hdf' % (save_filename_prefix, n_epochs)
    if path.isfile(gan_weights_file):
        # load InfoGAN weights that already existed
        print('loading InfoGAN model weights')
        infogan.load_weights(gan_weights_file)
    else:
        # train InfoGAN
        print('training InfoGAN model')
        train_infogan(x_train, generator, discriminator, infogan, losses,
                      batch_size=batch_size,
                      n_epochs=n_epochs,
                      save_every=n_save_every,
                      save_mode=save_mode,
                      save_filename_prefix=save_filename_prefix)
        infogan.save_weights(gan_weights_file)

    # accuracy test
    y_test_softmax = classifier.predict(x_test)
    y_test_pred = np.argmax(y_test_softmax, axis=1)
    acc_pred, w_cluster = cluster_acc(y_test, y_test_pred)
    print('Classifier accuracy: %.4f' % acc_pred)
    print(w_cluster.astype(int))

    # generate new data and save the plot
    if save_mode is 'multi_channel':
        # display generated images channel by channel
        for ic in range(n_ch):
            save_filename_image_gen = '%s_cat_gen_ch%d_epoch%d.pdf' % (save_filename_prefix, ic, n_epochs)
            plot_gen(generator, (n_class, 10), (15, 15), ic, True, save_filename_image_gen, cmap='gray')
    elif save_mode is 'color':
        save_filename_image_gen = '%s_cat_gen_epoch%d.pdf' % (save_filename_prefix, n_epochs)
        plot_gen(generator, (n_class, 10), (15, 15), 'color', True, save_filename_image_gen)

    pass