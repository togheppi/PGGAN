import torch
from torch.autograd import Variable
import torchvision.utils as vutils
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
# from edge_detector import edge_detect


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


# De-normalization
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# For logger
def to_np(x):
    return x.data.cpu().numpy()


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


# Plot losses
def plot_loss(avg_losses, num_epochs, save=False, save_dir='results/', show=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0, num_epochs)
    temp = 0.0
    for i in range(len(avg_losses)):
        temp = max(np.max(avg_losses[i]), temp)
    ax.set_ylim(0, temp*1.1)
    plt.xlabel('# of Epochs')
    plt.ylabel('Loss values')

    for i in range(len(avg_losses)):
        if i % 2 == 0:
            plt.plot(avg_losses[i], label='D_loss')
        else:
            plt.plot(avg_losses[i], label='G_loss')
    plt.legend()

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'Loss_values_epoch_{:d}'.format(num_epochs) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


# Make gif
def make_gif(model_name, num_epochs, save_dir='results/'):
    gen_image_plots = []
    for epoch in range(num_epochs):
        # plot for generating gif
        filename = save_dir + 'Result_%s_epoch_%d.png' % (model_name, epoch + 1)
        gen_image_plots.append(imageio.imread(filename))

    save_fn = save_dir + 'Result_%s_epoch_%d.gif' % (model_name, num_epochs)
    imageio.mimsave(save_fn, gen_image_plots, fps=5)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('ConvTranspose2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    # elif classname.find('Norm') != -1:
    #     m.weight.data.normal_(1.0, 0.02)
    #     if m.bias is not None:
    #         m.bias.data.zero_()


def result_images(tensor, epoch, edge=False, model_name='DCGAN', save_dir='./result', save=True,
                  nrow=5, padding=2, pad_value=0, normalize=True, range=(-1.0, 1.0)):

    tensor = tensor.cpu()
    grid = vutils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                            normalize=normalize, range=range, scale_each=False)
    ndarr = grid.mul(255).byte().permute(1, 2, 0).numpy()

    # grid1 = vutils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
    #                         normalize=normalize, range=None, scale_each=False)
    # grid1 = (grid1 - grid1.min()) / (grid1.max() - grid1.min())
    # ndarr1 = grid1.mul(255).byte().permute(1, 2, 0).numpy()

    if save:
        im = Image.fromarray(ndarr)
        filename = 'Result_%s_epoch_%d.png' % (model_name, epoch + 1)
        im.save(save_dir + filename)
        # im1 = Image.fromarray(ndarr1)
        # filename1 = 'Result2_%s_epoch_%d.png' % (model_name, epoch + 1)
        # im1.save(save_dir + filename1)

        # if edge:
        #     edge_arr = edge_detect(ndarr)
        #     edge_im = Image.fromarray(edge_arr)
        #     filename = 'Result_edge_%s_epoch_%d.png' % (model_name, epoch + 1)
        #     edge_im.save(save_dir + filename)

    return ndarr
