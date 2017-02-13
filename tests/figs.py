
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

SAVE_DIR = os.path.expanduser('~/Desktop/bolt/figs/')


def save_fig(name):
    plt.savefig(os.path.join(SAVE_DIR, name + '.pdf'))


def set_style():
    plt.style.use(['seaborn-white', 'seaborn-paper'])
    matplotlib.rc("font", family="Times New Roman")


def popcount_fig():
    bolt_128d_8b = np.random.randn(10) + 12
    bolt_128d_16b = np.random.randn(10) + 6
    bolt_128d_32b = np.random.randn(10) + 3
    popcnt_128d_8b = np.random.randn(10) + 10
    popcnt_128d_16b = np.random.randn(10) + 5
    popcnt_128d_32b = np.random.randn(10) + 2

    # bolt_512d_8b = np.random.randn(10) + 12 / 2
    # bolt_512d_16b = np.random.randn(10) + 6 / 2
    # bolt_512d_32b = np.random.randn(10) + 3 / 2
    # popcnt_512d_8b = np.random.randn(10) + 10 / 2
    # popcnt_512d_16b = np.random.randn(10) + 5 / 2
    # popcnt_512d_32b = np.random.randn(10) + 2 / 2

    # sb.set_context("poster", rc={"figure.figsize": (8, 4)})
    sb.set_context("talk", rc={"figure.figsize": (6, 4)})

    # sb.set_palette("Set1", n_colors=2)
    set_palette(ncolors=2)

    # fig, axes = plt.subplots(1, 2)

    dicts_128d = []
    dicts_128d += [{'algo': 'Bolt', 'x': '8B',  't': t} for t in bolt_128d_8b]
    dicts_128d += [{'algo': 'Bolt', 'x': '16B',  't': t} for t in bolt_128d_16b]
    dicts_128d += [{'algo': 'Bolt', 'x': '32B',  't': t} for t in bolt_128d_32b]
    dicts_128d += [{'algo': 'Binary Embedding', 'x': '8B',  't': t} for t in popcnt_128d_8b]
    dicts_128d += [{'algo': 'Binary Embedding', 'x': '16B',  't': t} for t in popcnt_128d_16b]
    dicts_128d += [{'algo': 'Binary Embedding', 'x': '32B',  't': t} for t in popcnt_128d_32b]

    df = pd.DataFrame.from_records(dicts_128d)
    print "df cols: ", df.columns
    # df.rename(columns={'algo': 'Algorithm'}, inplace=True)
    df.rename(columns={'algo': ' '}, inplace=True)  # hide from legend

    ax = sb.barplot(x='x', y='t', hue=' ', ci=95, data=df)
    ax.set_title('Distance Computations Per Second')
    ax.set_xlabel('Encoding Length (Bytes)')
    ax.set_ylabel('Millions of distances / sec')

    plt.tight_layout()
    save_fig('popcount_speed')
    # plt.show()


def encoding_fig(data_enc=True):
    # sb.set_context("talk", rc={"figure.figsize": (6, 6)})
    sb.set_context("talk", rc={"figure.figsize": (7, 7)})
    # sb.set_context("talk", rc={"figure.figsize": (8, 8)})
    # sb.set_context("talk", rc={"figure.figsize": (9, 9)})

    # fig, axes = plt.subplots(3, 1)
    fig, axes = plt.subplots(3, 2)

    ALGOS = ['Bolt', 'PQ', 'OPQ', 'PairQ']
    algo2offset = {'Bolt': 100, 'PQ': 50, 'OPQ': 30, 'PairQ': 25}
    lengths = [64, 128, 256, 512, 1024]
    # results_for_algos_lengths =

    # sb.set_palette("Set1", n_colors=len(ALGOS))
    set_palette(ncolors=len(ALGOS))

    condition = ALGOS

    # prefix = 'Data' if data_enc else 'Query'

    # ================================ data encoding

    # prefix = 'Data'

    # ------------------------ 8B encodings

    data = np.random.randn(1, len(lengths), len(algo2offset))
    for i, algo in enumerate(ALGOS):
        data[:, :, i] += algo2offset[algo]
    data /= np.arange(len(lengths)).reshape((1, -1, 1))
    # if not data_enc:  # query encoding
    #     data += np.random.randn(*data.shape) * 5

    ax = axes[0, 0]
    # sb.tsplot(data=data, condition=condition, time=lengths, ax=ax)
    sb.tsplot(data=data, condition=None, time=lengths, ax=ax)
    # ax.set_title(prefix + ' Encoding Speed, 8B codes')
    ax.set_title('Data Encoding Speed', y=1.02)

    # ------------------------ 16B encodings

    data /= 2
    ax = axes[1, 0]
    sb.tsplot(data=data, condition=None, time=lengths, ax=ax)
    # ax.set_title(prefix + ' Encoding Speed, 16B codes')

    # ------------------------ 32B encodings

    data /= 2
    ax = axes[2, 0]
    sb.tsplot(data=data, condition=None, time=lengths, ax=ax)
    # ax.set_title(prefix + ' Encoding Speed, 32B codes')

    # ================================ query encoding

    # prefix = 'Query'

    # fake data
    data *= 8
    data += np.random.randn(*data.shape) * 5

    # ------------------------ 8B encodings

    ax = axes[0, 1]
    sb.tsplot(data=data, condition=condition, time=lengths, ax=ax)
    # ax.set_title(prefix + ' Encoding Speed')
    ax.set_title('Query Encoding Speed', y=1.02)

    # ------------------------ 16B encodings

    data /= 2
    ax = axes[1, 1]
    sb.tsplot(data=data, condition=None, time=lengths, ax=ax)
    # ax.set_title(prefix + ' Encoding Speed, 16B codes')

    # ------------------------ 32B encodings

    data /= 2
    ax = axes[2, 1]
    sb.tsplot(data=data, condition=None, time=lengths, ax=ax)
    # ax.set_title(prefix + ' Encoding Speed, 32B codes')

    # ------------------------ postproc + save plot

    for ax in axes[-1, :].ravel():
        ax.set_xlabel('Vector length')
    for ax in axes[:, 0]:
        ax.set_ylabel('Million vectors / s')

    # only bottom row gets xlabels
    for ax in axes[:2, :].ravel():
        plt.setp(ax.get_xticklabels(), visible=False)

    # show byte counts on the right
    fmt_str = "{}B Encodings"
    for i, ax in enumerate(axes[:, 1].ravel()):
        ax.yaxis.set_label_position('right')
        ax.set_ylabel(fmt_str.format((2 ** i) * 8), labelpad=10, fontsize=15)

    plt.tight_layout()
    save_fig('encoding_speed')
    # plt.show()
    # if data_enc:
    #     save_fig('encoding_speed_data')
    # else:
    #     save_fig('encoding_speed_query')
    # plt.show()


def query_speed_fig():
    # experiment params: fixed N = 100k, D = 256, Q = 1024;
    # layout: rows = 8B, 16B, 32B; bar graph in each row
    #   alternative: plot in each row vs batch size
    # algos: Bolt; PQ; OPQ; PairQ; Matmul, batch={1, 16, 64, 256}

    sb.set_context("talk", rc={"figure.figsize": (6, 8)})

    fig, axes = plt.subplots(3, 1)

    ALGOS = ['Bolt', 'PQ', 'OPQ', 'PairQ',
             # 'Matmul Batch 1', 'Matmul Batch 16', 'Matmul Batch 64', 'Matmul Batch 256']
             # 'Matmul Batch1', 'Matmul Batch16', 'Matmul Batch64', 'Matmul Batch256']
             'Matmul 1', 'Matmul 16', 'Matmul 64', 'Matmul 256']
    algo2offset = {'Bolt': 100, 'PQ': 50, 'OPQ': 30, 'PairQ': 25,
                   # 'Matmul Batch 1': 1, 'Matmul Batch 16': 16,
                   # 'Matmul Batch 64': 64, 'Matmul Batch 256': 256}
                   # 'Matmul Batch1': 1, 'Matmul Batch16': 16,
                   # 'Matmul Batch64': 64, 'Matmul Batch256': 256}
                   'Matmul 1': 1, 'Matmul 16': 16, 'Matmul 64': 64,
                   'Matmul 256': 256}

    # sb.set_palette("Set1", n_colors=len(ALGOS))
    set_palette(ncolors=len(ALGOS))

    for i, nbytes in enumerate([8, 16, 32]):
        bytes_str = '{}B'.format(nbytes)
        dicts = []
        for algo in ALGOS:
            dps = np.random.randn(10) + 256 / nbytes
            dps += algo2offset[algo] / nbytes
            dicts += [{'algo': algo, 'x': bytes_str, 'y': y} for y in dps]

        df = pd.DataFrame.from_records(dicts)
        print "df cols: ", df.columns
        df.rename(columns={'algo': ' '}, inplace=True)  # hide from legend

        # ax = sb.barplot(x='x', y='y', hue=' ', ci=95, data=df, ax=axes[i])
        ax = sb.barplot(x='x', y='y', hue=' ', hue_order=ALGOS, ci=95, data=df, ax=axes[i])

    for ax in axes[:-1]:
        # remove x labels except for bottom axis
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.get_xaxis().set_visible(False)

    # ------------------------ clean up / format axes

    # print axes[-1].get_xticks()
    # print axes[-1].get_xlim()
    # start = -.5
    # end = .5
    # start = -.5 + 1. / (len(ALGOS))
    end = .5 * (len(ALGOS) / float((len(ALGOS) + 2)))
    start = -end
    # start = -.5 + 1. / (len(ALGOS) + 1) + .2
    # end = .5 - 1. / (len(ALGOS) + 1) + .2
    # tick_positions = np.linspace(start + .05, end - .05, len(ALGOS))
    tick_positions = np.linspace(start + .04, end - .07, len(ALGOS))
    # print tick_positions
    # return

    for ax in axes:
        ax.set_xlim([start - .02, end + .02])
        ax.set_ylabel('Millions of Distances/s')
        ax.legend_.remove()

    # add byte counts on the right
    sb.set_style("white")  # adds border (spines) we have to remove
    # sb.set_style("ticks")
    fmt_str = "{}B Encodings"
    for i, ax in enumerate(axes):
        ax2 = ax.twinx()
        sb.despine(ax=ax2, top=True, left=True, bottom=True, right=True)
        ax2.get_xaxis().set_visible(False)
        # ax2.get_yaxis().set_visible(False)  # nope, removes ylabel
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.yaxis.set_label_position('right')
        ax2.set_ylabel(fmt_str.format((2 ** i) * 8), labelpad=10, fontsize=15)

    # ------------------------ have bottom / top axes print title, x info

    axes[0].set_title('Query Speed', y=1.02)

    axes[-1].set_xticks(tick_positions)
    # xlabels = ["\n".join(name.split(' ')) for name in ALGOS]
    # xlabels = ["\nBatch".join(name.split(' Batch')) for name in ALGOS]
    xlabels = ALGOS
    axes[-1].set_xticklabels(xlabels, rotation=70)
    axes[-1].set_xlabel("", labelpad=-20)
    # plt.setp(axes[-1].get_xlabel(), visible=False)  # doesn't work

    # ------------------------ show / save plot

    plt.tight_layout()
    save_fig('query_speed')
    # plt.show()


def set_palette(ncolors=8):
    pal = sb.color_palette("Set1", n_colors=8)
    # pal2 = sb.xkcd_palette(["amber"])
    # # pal2 = sb.color_palette(["yellow"])
    # pal[5] = pal[0]
    # pal[0] = pal2[0]
    sb.set_palette(pal)

    return pal


def main():
    # pal = set_palette()
    # sb.palplot(pal)
    # plt.show()

    # popcount_fig()
    # encoding_fig(data_enc=True)
    # encoding_fig(data_enc=False)
    # encoding_fig()
    query_speed_fig()


if __name__ == '__main__':
    main()
