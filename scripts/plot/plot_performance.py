"""
Plot results as a clustered bar graph.
"""
import os
import argparse
import sys
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
sys.path.insert(0, here + '/../../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def set_size(width, fraction=1, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    """
    golden_ratio = 1.618
    height = (width / golden_ratio) * (subplots[0] / subplots[1])
    return width, height


def organize_results(args, df, feature_type, test_type, methods):
    """
    Put results into dataset clusters.
    """
    results = []
    std_list = []
    n_datasets = 0

    # filter setting
    temp = df[df['feature_type'] == feature_type]
    temp = temp[temp['test_type'] == test_type]

    method_list = []

    for dataset in args.dataset:
        dataset_std_list = []

        result = {'dataset': dataset}

        temp2 = temp[temp['dataset'] == dataset]

        if len(temp2) == 0:
            continue

        method_list = []
        n_datasets += 1

        # add methods
        for sgl_method, sgl_stacks, pgm in methods:
            temp3 = temp2[temp2['sgl_method'] == sgl_method]
            temp3 = temp3[temp3['sgl_stacks'] == sgl_stacks]
            temp3 = temp3[temp3['pgm'] == pgm]

            key = '{}_{}_{}'.format(sgl_method, sgl_stacks, pgm)
            metric_diff = temp3['{}_diff_mean'.format(args.metric)].values[0]
            metric_std = temp3['{}_diff_std'.format(args.metric)].values[0]

            result[key] = metric_diff * 100
            dataset_std_list.append(metric_std * 100)
            method_list.append(key)

        std_list += dataset_std_list + dataset_std_list

        results.append(result)
    res_df = pd.DataFrame(results)

    return res_df, method_list, std_list, n_datasets


def main(args):
    print(args)

    settings = [('full', 'full'), ('full', 'inductive'),
                ('limited', 'full'), ('limited', 'inductive')]

    methods = [('holdout', 1, 'None'), ('holdout', 2, 'None'),
               ('cv', 1, 'None'), ('cv', 2, 'None'),
               ('None', 0, 'mrf'),
               ('holdout', 1, 'mrf'), ('holdout', 2, 'mrf'),
               ('cv', 1, 'mrf'), ('cv', 2, 'mrf')]

    labels = ['Holdout (1)', 'Holdout (2)', 'CV (1)', 'CV (2)', 'MRF only',
              'Holdout (1) + MRF', 'Holdout (2) + MRF', 'CV (1) + MRF', 'CV (2) + MRF']

    # matplotlib settings
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)
    plt.rc('axes', labelsize=17)
    plt.rc('axes', titlesize=17)
    plt.rc('legend', fontsize=11)
    plt.rc('legend', title_fontsize=9)
    plt.rc('lines', linewidth=2)
    plt.rc('lines', markersize=5)

    # setup figure
    width = 5
    width, height = set_size(width=width * 3, fraction=1, subplots=(len(settings), 3))
    fig, axs = plt.subplots(len(settings), 1, figsize=(width, height * 0.93), sharex=True, sharey=True)

    colors = ['0.5', '0.25', '0.5', '0.25', '1.0', '0.5', '0.25', '0.5', '0.25']
    hatches = ['-', '+', 'x', '\\', '*', 'o', 'O', '.', '/']

    # get results
    main_fp = os.path.join(args.in_dir, 'results.csv')
    main_df = pd.read_csv(main_fp)

    for i, (feature_type, test_type) in enumerate(settings):
        ax = axs[i]

        res_df, keys, std_list, n_datasets = organize_results(args, main_df,
                                                              feature_type=feature_type,
                                                              test_type=test_type,
                                                              methods=methods)

        yerr = np.reshape(std_list, (len(methods), 2, n_datasets), order='F')

        res_df.plot(x='dataset', y=keys, yerr=yerr, kind='bar', hatch=hatches,
                    color=colors, ax=axs[i], edgecolor='k', linewidth=0.5, capsize=2)

        bars = ax.patches
        hatch_cnt = 0
        hatch_ndx = -1
        for j in range(len(bars)):
            if hatch_cnt == n_datasets:
                hatch_ndx += 1
                hatch_cnt = 0
            bars[j].set_hatch(hatches[hatch_ndx])
            hatch_cnt += 1

        test_label = 'Inductive + Transductive' if test_type == 'full' else 'Inductive'
        ax.set_title('Feature: {}, Test samples: {}'.format(feature_type.capitalize(), test_label), loc='left')
        ax.grid(which='major', axis='y')
        ax.set_axisbelow(True)
        ax.set_ylabel(r'Test {} $\Delta$ (%)'.format(args.metric))
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=0)

        if i == 0:
            leg = ax.legend(labels=labels, ncol=1, framealpha=1.0, loc='upper left')

        if i > 0:
            ax.get_legend().remove()

    # Get the bounding box of the original legend
    bb = leg.get_bbox_to_anchor().inverse_transformed(axs[0].transAxes)

    # Change to location of the legend.
    yOffset = 0.015
    bb.y0 += yOffset
    bb.y1 += yOffset
    leg.set_bbox_to_anchor(bb, transform=axs[0].transAxes)

    out_dir = os.path.join(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    fig.tight_layout()
    fp = os.path.join(out_dir, 'peformance_{}.pdf'.format(args.metric))
    plt.savefig(fp)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, nargs='+', help='datasets to use for plotting',
                        default=['youtube', 'twitter', 'soundcloud'])
    parser.add_argument('--in_dir', type=str, default='output/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/performance/', help='output directory.')
    parser.add_argument('--metric', type=str, default='auc', help='performance metric.')
    args = parser.parse_args()
    main(args)
