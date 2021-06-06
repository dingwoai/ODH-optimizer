import os
import matplotlib.pyplot as plt
import numpy as np


def plot_single(data, text, savename):    
    iter = np.arange(0, len(data), 1)
    fig = plt.figure()
    ax = fig.add_subplot(111) 
    ax.set_yscale("log")
    ax.plot(iter, data, label=text)
    ax.legend(loc='upper right', prop={"family": 'Microsoft YaHei'}, fontsize=10)
    # plt.show()
    plt.savefig(os.path.join("./assets/", savename))


def plot_multi(datas, texts, savename):
    nb_data = len(datas)
    assert nb_data==len(texts), f'length of data {nb_data} not equal to length of texts {len(texts)}'
    fig = plt.figure()
    ax = fig.add_subplot(111) 
    for i in range(nb_data):
        iter = np.arange(0, len(datas[i]), 1)
        ax.set_yscale("log")
        ax.plot(iter, datas[i], label=texts[i])
        ax.legend(loc='upper right', prop={"family": 'Microsoft YaHei'}, fontsize=10)
    # plt.show()
    plt.savefig(os.path.join("./assets/", savename))


def plot_from_npz(root_dir, filenames, savename):
    data_error, data_alpha, data_gnorm = [], [], []
    for f in filenames:
        print(f)
        data = np.load(os.path.join(root_dir, f+ '.npz'), allow_pickle=True)
        data_error.append(data['error'])
        data_alpha.append(data['alpha'])
        data_gnorm.append(data['gk_norm'])
    plot_multi(data_error, filenames, 'error-'+savename)
    plot_multi(data_alpha, filenames, 'alpha-'+savename)
    plot_multi(data_gnorm, filenames, 'gnorm-'+savename)


if __name__=='__main__':
    root_dir = './assets/'
    filenames = ['ODH1-exp5', 'ODH2-exp5', 'BB1-exp5', 'BB2-exp5']
    plot_from_npz(root_dir, filenames, savename='exp5-2.png')