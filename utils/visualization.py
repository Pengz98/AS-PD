import os.path

import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import datetime
import matplotlib.ticker as ticker

timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = Path(BASE_DIR)
OUTPUT_PATH = OUTPUT_PATH.joinpath('output')
OUTPUT_PATH.mkdir(exist_ok=True)
OUTPUT_PATH_TIME = OUTPUT_PATH.joinpath(timestr)


def visu_pc_w_dis(pc_dis, window_name='default', return_pcd=False):
    '''
    :param pc_dis: [N,3+1]
    :return: visu
    '''
    if pc_dis.shape[1] == 3:
        dis = np.ones((pc_dis.shape[0], 1))
        pc_dis = np.concatenate([pc_dis, dis], -1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_dis[:, :3])
    max_label = pc_dis[:, -1].max()
    colors = plt.get_cmap("tab20")(pc_dis[:, -1] / 3 * (max_label if max_label > 0 else 1))
    pcd.colors = o3d.utility.Vector3dVector(colors[:,:3])

    if return_pcd:
        return pcd
    else:
        o3d.visualization.draw_geometries([pcd], window_name=window_name)



def savetxt_pc_w_dis(pc_dis, filename='default', dirname=None):
    '''
    :param pc_dis: [N,3+1]
    :return: txt
    '''
    if pc_dis.shape[1] == 3:
        dis = np.ones((pc_dis.shape[0], 1))
        pc_dis = np.concatenate([pc_dis, dis], -1)
    if dirname is not None:
        dir_path = OUTPUT_PATH.joinpath(dirname)
        dir_path.mkdir(exist_ok=True)
        txt_path = str(dir_path) + '/' + str(filename)+'.txt'
    else:
        OUTPUT_PATH_TIME.mkdir(exist_ok=True)
        txt_path = str(OUTPUT_PATH_TIME) + '/' + str(filename) + '.txt'

    np.savetxt(txt_path, pc_dis, fmt='%.5f, %.5f, %.5f, %.8f', delimiter='\n')


def plot_line_chart(x_value, y_value, line_label='line_a',
                    x_value1=None, y_value1=None, line_label1='line_b',
                    x_value2=None, y_value2=None, line_label2='line_c',
                    x_value3=None, y_value3=None, line_label3='line_d',
                    x_value4=None, y_value4=None, line_label4='line_e',
                    x_value5=None, y_value5=None, line_label5='line_f',
                    title=None, save_fig=False, x_label='x', y_label='y',
                    reverse_x=False, log_x=False, save_name='default', font_size=15):

    plt.figure()

    plt.plot(x_value, y_value, label=line_label)

    if x_value1 is not None and y_value1 is not None:
        # plt.plot(x_value1, y_value1, label=line_label1, color='tab:blue', linestyle='-.')
        plt.plot(x_value1, y_value1, label=line_label1)

    if x_value2 is not None and y_value2 is not None:
        plt.plot(x_value2, y_value2, label=line_label2)

    if x_value3 is not None and y_value3 is not None:
        # plt.plot(x_value3, y_value3, label=line_label3, color='tab:orange', linestyle='-.')
        plt.plot(x_value3, y_value3, label=line_label3)

    if x_value4 is not None and y_value4 is not None:
        plt.plot(x_value4, y_value4, label=line_label4)

    if x_value5 is not None and y_value5 is not None:
        plt.plot(x_value5, y_value5, label=line_label5)

    if title is not None:
        plt.title(title)
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)

    plt.legend(fontsize=font_size)

    if reverse_x:
        ax = plt.gca()
        ax.invert_xaxis()
        # ax.set_xscale('log')

    if log_x:
        ax = plt.gca()
        ax.set_xscale("log", base=2)
        # x_range = np.power(2, np.arange(5,11))
        # plt.xticks(x_range, ('32','64','128','256','512','1024'), fontsize=15)
        x_range = np.power(2, np.arange(4,11))
        plt.xticks(x_range, ('16','32','64','128','256','512','1024'), fontsize=font_size)
        plt.yticks(fontsize=font_size)
    else:
        x_range = np.linspace(500,2000,4)
        plt.xticks(x_range, fontsize=font_size)
        plt.yticks(fontsize=font_size)

        # x_range = np.power(2, np.arange(4,11))
        # ax.set_xticks(x_range)
        # ax.xaxis.set_major_locator(x_range)

    plt.grid()

    plt.tight_layout()

    if save_fig:
        plt.savefig('../images/%s.pdf' % str(save_name), dpi=600, format='pdf')

    plt.show()

def plot_double_y_line_chart(x_value, y_value, line_label,
                             x_value1, y_value1, line_label1,
                             x1_value, y1_value, line1_label,
                             x1_value1, y1_value1, line1_label1,
                             x_value2=None, y_value2=None, line_label2=None,
                             title='line chart', save_fig=False, x_label='x', y_label='y',y_label1='y1',
                             reverse_x=False, log_x=False, save_name=None):

    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.plot(x_value, y_value, label=line_label)
    ax.plot(x_value1, y_value1, label=line_label1)

    if x_value2 is not None:
        ax.plot(x_value2, y_value2, label=line_label2)


    ax2 = ax.twinx()
    ax2.plot(x1_value, y1_value, label=line1_label, linestyle='--')
    ax2.plot(x1_value1, y1_value1, label=line1_label1, linestyle='--')


    # if x_value2 is not None and y_value2 is not None:
    #     plt.plot(x_value2, y_value2, label=line_label2)
    #
    # if x_value3 is not None and y_value3 is not None:
    #     plt.plot(x_value3, y_value3, label=line_label3)
    #
    # if x_value4 is not None and y_value4 is not None:
    #     plt.plot(x_value4, y_value4, label=line_label4)
    #
    # if x_value5 is not None and y_value5 is not None:
    #     plt.plot(x_value5, y_value5, label=line_label5)

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(x_label, fontsize=15)
    ax.set_ylabel(y_label, fontsize=15)
    ax2.set_ylabel(y_label1, fontsize=15)

    ax.legend(loc=5, bbox_to_anchor=(1,0.4), fontsize=15)    # 2：upper left, 6: mid left
    ax2.legend(loc=4, bbox_to_anchor=(1,0.1), borderpad=0.3, fontsize=15)   # 1: upper right, 5: mid right, 4: down right
    # ax.legend(loc=2, fontsize=15)  # 2：upper left, 6: mid left
    # ax2.legend(loc=1, fontsize=15)  # 1: upper right, 5: mid right, 4: down right

    if reverse_x:
        ax = plt.gca()
        ax.invert_xaxis()
        # ax.set_xscale('log')

    if log_x:

        ax = plt.gca()
        ax.set_xscale("log", base=2)

        x_range = np.power(2, np.arange(5,10))
        plt.xticks(x_range, ('32','64','128','256','512'), fontsize=15)
        plt.yticks(fontsize=15)
        # ax.set_xticks(x_range)
        # ax.xaxis.set_major_locator(x_range)

    ax.grid()

    plt.tight_layout()

    if save_fig:
        plt.savefig('../images/%s.pdf' % str(save_name), dpi=800, format='pdf')

    plt.show()


def plot_3d_point_cloud(x, y, z, show=True, show_axis=True, in_u_sphere=True, marker='.', c='b', s=8, alpha=.8,
                        figsize=(5, 5), elev=10, azim=240, miv=None, mav=None, squeeze=0.7, axis=None, title=None,
                        *args, **kwargs):
    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    sc = ax.scatter(x, y, z, marker=marker, c=c, s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    if in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
        miv = -0.5
        mav = 0.5
    else:
        if miv is None:
            miv = squeeze * np.min([np.min(x), np.min(y), np.min(z)])  # Multiply with 'squeeze' to squeeze free-space.
        if mav is None:
            mav = squeeze * np.max([np.max(x), np.max(y), np.max(z)])
        ax.set_xlim(miv, mav)
        ax.set_ylim(miv, mav)
        ax.set_zlim(miv, mav)
        plt.tight_layout()

    if not show_axis:
        plt.axis('off')

    if 'c' in kwargs:
        plt.colorbar(sc)

    if show:
        plt.show()

    return fig, miv, mav


if __name__ == '__main__':

    vi_aspd_32 = np.load('../result/varyinput/varyinput_32.npy')
    vi_aspd_256 = np.load('../result/varyinput/varyinput_256.npy')
    vi_fps_32 = np.load('../result/varyinput/varyinput_fps_32.npy')
    vi_fps_256 = np.load('../result/varyinput/varyinput_fps_256.npy')
    vi_rs_32 = np.load('../result/varyinput/varyinput_rs_32.npy')
    vi_rs_256 = np.load('../result/varyinput/varyinput_rs_256.npy')
    n_input_list = vi_aspd_32[:, 0].astype('int')

    # vary input 32
    plot_line_chart(x_value=n_input_list, y_value=vi_aspd_32[:, 1], line_label='AS-PD',
                    x_value1=n_input_list, y_value1=vi_fps_32[:, 1], line_label1='FPS',
                    x_value2=n_input_list, y_value2=vi_rs_32[:, 1], line_label2='RS',
                    log_x=False, title=None, save_fig=True, save_name='vary_cls_32',
                    x_label='Input Size', y_label='Classification Accuracy (%)', font_size=21)

    # vary input 256
    plot_line_chart(x_value=n_input_list, y_value=vi_aspd_256[:, 1], line_label='AS-PD',
                    x_value1=n_input_list, y_value1=vi_fps_256[:, 1], line_label1='FPS',
                    x_value2=n_input_list, y_value2=vi_rs_256[:, 1], line_label2='RS',
                    log_x=False, title=None, save_fig=True, save_name='vary_cls_256',
                    x_label='Input Size', y_label='Classification Accuracy (%)', font_size=21)


    full_pipeline = np.load('../result/new/full_pipeline.npy')
    ab_de = np.load('../result/new/ab_de.npy')
    ab_ps = np.load('../result/new/ab_ps.npy')
    ab_ts = np.load('../result/new/ab_ts.npy')

    snp = np.load('../result/new/snp.npy')
    pn = np.load('../result/new/pn.npy')
    fps = np.load('../result/new/fps.npy')
    rs = np.load('../result/new/rs.npy')

    n_sample_list = full_pipeline[:,0].astype('int')


    # ablation study acc
    plot_line_chart(x_value=n_sample_list, y_value=full_pipeline[:, 1], line_label='AS-PD',
                    x_value1=n_sample_list, y_value1=ab_de[:, 1], line_label1='AS-PD-noDA',
                    x_value2=n_sample_list, y_value2=ab_ps[:, 1], line_label2='AS-PD-RS',
                    x_value3=n_sample_list, y_value3=ab_ts[:, 1], line_label3='AS-PD-one_stage',
                    x_value4=n_sample_list, y_value4=fps[:, 1], line_label4='FPS',
                    log_x=True, title=None, save_fig=True, save_name='ab_cls_acc',
                    x_label='Sample Size', y_label='Classification Accuracy (%)', font_size=15)

    # classification acc
    plot_line_chart(x_value=n_sample_list, y_value=full_pipeline[:, 1], line_label='AS-PD',
                    x_value1=n_sample_list, y_value1=snp[:, 1], line_label1='SNP',
                    x_value2=n_sample_list, y_value2=pn[:, 1], line_label2='PN',
                    x_value3=n_sample_list, y_value3=fps[:, 1], line_label3='FPS',
                    x_value4=n_sample_list, y_value4=rs[:, 1], line_label4='RS',
                    log_x=True, title=None, save_fig=True, save_name='cls_acc',
                    x_label='Sample Size', y_label='Classification Accuracy (%)', font_size=21)

    # classification hd
    plot_line_chart(x_value=n_sample_list, y_value=0.01*full_pipeline[:, 3], line_label='AS-PD',
                    x_value1=n_sample_list, y_value1=0.01 * snp[:, 3], line_label1='SNP',
                    x_value2=n_sample_list, y_value2=0.01 * pn[:, 3], line_label2='PN',
                    x_value3=n_sample_list, y_value3=0.01 * fps[:, 3], line_label3='FPS',
                    x_value4=n_sample_list, y_value4=0.01 * rs[:, 3], line_label4='RS',
                    log_x=True, title=None, save_fig=True, save_name='cls_hd',
                    x_label='Sample Size', y_label='Hausdorff Distance', font_size=21)


    # registration
    regis_full_pipeline = np.load('../result/new/regis_ASPD.npy')
    regis_snp = np.load('../result/new/regis_SNP.npy')
    regis_fps = np.load('../result/new/regis_FPS.npy')
    regis_rs = np.load('../result/new/regis_RS.npy')

    n_sample_list = regis_full_pipeline[:,0].astype('int')

    plot_line_chart(x_value=n_sample_list, y_value=regis_full_pipeline[:, 1], line_label='AS-PD',
                    x_value1=n_sample_list, y_value1=regis_fps[:, 1], line_label1='FPS',
                    x_value2=n_sample_list, y_value2=regis_rs[:, 1], line_label2='RS',
                    x_value3=n_sample_list, y_value3=regis_snp[:, 1], line_label3='SNP',
                    log_x=True, title=None, save_fig=True, save_name='regis_mre',
                    x_label='Sample Size', y_label='Mean Rotation Error (DEG)', font_size=21)

    plot_line_chart(x_value=n_sample_list, y_value=regis_full_pipeline[:, 2], line_label='AS-PD',
                    x_value1=n_sample_list, y_value1=regis_fps[:, 2], line_label1='FPS',
                    x_value2=n_sample_list, y_value2=regis_rs[:, 2], line_label2='RS',
                    x_value3=n_sample_list, y_value3=regis_snp[:, 2], line_label3='SNP',
                    log_x=True, title=None, save_fig=True, save_name='regis_hd',
                    x_label='Sample Size', y_label='Hausdorff Distance', font_size=21)




