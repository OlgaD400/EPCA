"""Functions for generating figures."""

from typing import List, Optional, Tuple
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from EPCA.EPCA import EPCA
from EPCA.helper_functions import (
    match_components,
    run_epca,
    run_pca,
    run_rpca,
)


def plot_components(components: np.ndarray, imsize: Optional[Tuple] = None):
    """
    Visualize PCA components:

    Args:
    components (np.ndarray): The PCA components to visualize
    imsize(Optional[Tuple]): The shape of the component to be plotted. If none, the component is 1D.
    """
    for component in components:
        if imsize is None:
            plt.figure()
            plt.plot(component)
        else:
            plt.figure()
            plt.imshow(component.reshape(imsize))


def plot_epca_trials(
    epca_args: dict,
    num_trials: int,
    input_data: np.ndarray,
    true_components: np.ndarray,
    root_filename: str,
    figsize=(15, 10),
    show_outliers=True,
):
    """
    Plot num_components true vs predicted components for num_trials of EPCA.

    Plot boxplots of explained variance for each of the trials.

    Args:
        epca_args (dict): Dictionary of arguments for EPCA.
        num_components (int): Number of components for EPCA.
        num_trials (int): Number of trials to run.
        input_data (int): Data on which to run EPCA.
        true_components (np.ndarray): True components.
        root_filename (str): Path at which to save image. To be appended with "_epca_components_trials.pdf"
            or "_epca_variance_trials.pdf"
        figsize (Optional[Tuple]): size of figure
        show_outliers (bool): Whether or not to show outliers in the boxplots
            of explained variance.
    Returns:
        fig: Components figure.
        axs: Axes associated with the compnents figure.
        fig2: Variance figure.

    """
    num_components = epca_args.get("num_components")

    fig, axs = plt.subplots(
        num_components, num_trials, figsize=figsize, sharex=True, sharey=True
    )

    pc_data = {"run": [], "value": [], "pc": []}

    for trial in range(num_trials):
        epca = EPCA(
            num_components=num_components,
            num_bags=epca_args.get("num_bags", 100),
            bag_size=epca_args.get("bag_size", 10),
        )
        _, data_dimension = input_data.shape
        epca.run_epca(input_data)
        pc_map = match_components(true_components, epca.centers)

        for index in range(num_components):
            ind = np.where(epca.labels == pc_map[index])
            # for vector in epca.signed_vectors[ind]:
            #     axs[index, trial].plot(vector, c="gray")
            axs[index, trial].plot(true_components[index], c="r")
            axs[index, trial].plot(epca.centers[pc_map[index]], "--b")

            lower_limit = np.percentile(epca.signed_vectors[ind], 5, axis=0)
            upper_limit = np.percentile(epca.signed_vectors[ind], 95, axis=0)
            axs[index, trial].fill_between(
                np.arange(data_dimension), lower_limit, upper_limit, color="grey"
            )

            for value in epca.explained_variance[ind]:
                pc_data["run"].append("Run " + str(trial + 1))
                pc_data["value"].append(value)
                pc_data["pc"].append("PC" + str(index + 1))

    axs[0, 0].set_title("Run 1", fontsize=20)
    axs[0, 1].set_title("Run 2", fontsize=20)
    axs[0, 2].set_title("Run 3", fontsize=20)
    fig.suptitle("Runs of EPCA", fontsize=20)

    axs[0, 0].set_ylabel("PC 1", fontsize=20)
    axs[1, 0].set_ylabel("PC 2", fontsize=20)

    axs[0, 0].tick_params(axis="y", which="major", labelsize=16)
    axs[1, 0].tick_params(axis="y", which="major", labelsize=16)
    axs[1, 0].tick_params(axis="x", which="major", labelsize=16)
    axs[1, 1].tick_params(axis="x", which="major", labelsize=16)
    axs[1, 2].tick_params(axis="x", which="major", labelsize=16)

    legend_elements = [
        Line2D([0], [0], color="r", label="True Component"),
        Line2D([0], [0], color="b", label="Avg. Predicted Component", linestyle="--"),
        mpatches.Patch(color="grey", label="95% CI"),
    ]
    axs[0, 2].legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=16,
        framealpha=1,
    )

    plt.savefig(
        root_filename + "_epca_components_trials.pdf", format="pdf", bbox_inches="tight"
    )

    plt.figure(figsize=(10, 5))
    fig2 = sns.boxplot(
        data=pc_data, x="run", y="value", hue="pc", showfliers=show_outliers
    )
    fig2.set_title("Explained Variance of PCs per Run", size=16)
    fig2.set_ylabel("Explained Variance", size=16)
    fig2.tick_params(labelsize=16)
    fig2.legend(fontsize=16)

    plt.savefig(
        root_filename + "_epca_variance_trials.pdf", format="pdf", bbox_inches="tight"
    )

    return fig, axs, fig2


def plot_compare_methods(
    num_components: int,
    input_data: np.ndarray,
    true_components: np.ndarray,
    epca_args: Optional[dict] = None,
    pca_args: Optional[dict] = None,
    rpca_args: Optional[dict] = None,
):
    """
    Plot top two components of EPCA, RPCA, and PCA vs true components.

    Args:
        num_components (int): Num components to calculate.
        epca_args (dict): Dictionary of arguments for EPCA.
        pca_args (dict): Dictionary of arguments for PCA.
        rpca_args (dict): Dictionary of arguments for RPCA.
        input_data (dict): Data on which to run PCA.
        true_components (np.ndarray): True components.
    Return:
        fig: Figure comparing the top two components as predicted by up to three methods.
    """
    if epca_args is not None:
        ####### EPCA ###############
        epca_pred_components, _, _ = run_epca(
            data=input_data,
            num_components=num_components,
            num_bags=epca_args.get("num_bags", 100),
            bag_size=epca_args.get("bag_size", 10),
        )
        epca_map = match_components(true_components, epca_pred_components)

    ########## PCA ##########
    if pca_args is not None:
        final_pca, _, _ = run_pca(
            images=input_data,
            num_components=num_components,
        )
        pca_map = match_components(true_components, final_pca)

    ######## RPCA ##############
    if rpca_args is not None:
        final_rpca, _, _ = run_rpca(
            images=input_data,
            num_components=num_components,
            reg_E=rpca_args.get("reg_E", 1.0),
            reg_J=rpca_args.get("reg_J", 1.0),
            learning_rate=rpca_args.get("learning_rate", 1.1),
            n_iter_max=rpca_args.get("n_iter_max", 50),
        )
        rpca_map = match_components(true_components, final_rpca)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

    for index in range(num_components):
        axs[index].plot(true_components[index], c="r", label="True")
        axs[index].set_title("PC " + str(index + 1))
        if epca_args is not None:
            axs[index].plot(
                epca_pred_components[epca_map[index]],
                c="b",
                linestyle="dashed",
                label="EPCA",
            )
        if pca_args is not None:
            axs[index].plot(
                final_pca[pca_map[index]],
                c="g",
                label="PCA",
                linestyle="dotted",
                linewidth=3,
            )
        if rpca_args is not None:
            axs[index].plot(
                final_rpca[rpca_map[index]], c="k", linestyle="dashdot", label="RPCA"
            )
        axs[index].legend()
        axs[1].yaxis.set_tick_params(labelbottom=True)

    return fig


def plot_varied_levels_noise(
    first_component_performance: List[np.ndarray],
    second_component_performance: List[np.ndarray],
    feature_1: List,
    feature_2: List,
    subfigure_title: str,
    subfigure_caption: str,
    suptitle: str,
    filepath: str,
    **kwargs
):
    """
    Make and save varied noise level plots.

    Args:
        first_component_performance (List[np.ndarray]): Performance of first components. Must be in order pca, rpca, epca.
        second_component_performance (List[np.ndarray]): Performance of second components. Must be in order pca, rpca, epca.
        feature_1 (List): List of values that will determine the number of columns in the plot.
            Will also be used with "subfigure title" to title the columns.
        feature_2 (List): List of values that will determine the number of markers in a subfigure.
        subfigure_title (str): Title of each subfigure. "subfigure_title" + ' ' + feature_1[column] denotes a column's title.
        subfigure_caption (str): x label of each subfigure column
        suptitle (str): Title of the entire figure
        filepath (str): Filepath to save the figure to. Do not enclode file extension. Automatically saved as eps.

    Kwargs:
        title_size (float): Size of suptitle.
        text_size (float): Size of text in axis labels
        marker_size (float): Plot marker size
        axis_pad (float): Axis padding.
        figsize (Tuple[int]): Figure size.
        y1_step (float): Step size of y axis labels for first row.
        y2_step (float): Step size of y axis labels for second row.

    Returns:
        fig: Figure
        axs: Axes associated with the figure.
    """
    title_size = kwargs.get("title_size", 30)
    text_size = kwargs.get("text_size", 25)
    marker_size = kwargs.get("marker_size", 10)
    axis_pad = kwargs.get("axis_pad", 20)
    figsize = kwargs.get("figsize", (20, 10))
    y1_step = kwargs.get("y1_step", 10)
    y2_step = kwargs.get("y2_step", 20)

    pca_1, rpca_1, epca_1 = first_component_performance
    pca_2, rpca_2, epca_2 = second_component_performance

    max_y1 = int(np.max(np.concatenate(first_component_performance)) + 1)
    max_y2 = int(np.max(np.concatenate(second_component_performance)) + 1)

    fig, axs = plt.subplots(2, len(feature_1), figsize=figsize, sharey="row")
    locations = np.arange(len(feature_2))

    axs[0, 0].set_ylabel("Avg. % Relative Error", size=text_size, labelpad=axis_pad)
    axs[1, 0].set_ylabel("Avg. % Relative Error", size=text_size, labelpad=axis_pad)

    num_markers = len(feature_2)
    num_cols = len(feature_1)

    for index in range(num_cols):
        axs[0, index].plot(
            locations,
            epca_1[index * num_markers : index * num_markers + num_markers],
            "-o",
            c="tab:orange",
            markersize=marker_size,
        )

        axs[0, index].plot(
            locations,
            pca_1[index * num_markers : index * num_markers + num_markers],
            "x-",
            c="tab:blue",
            markersize=marker_size,
        )

        axs[0, index].plot(
            locations,
            rpca_1[index * num_markers : index * num_markers + num_markers],
            "-v",
            c="tab:green",
            markersize=marker_size,
        )

        axs[0, index].set_title(
            subfigure_title + " " + str(feature_1[index]), size=text_size
        )

        axs[1, index].plot(
            locations,
            epca_2[index * num_markers : index * num_markers + num_markers],
            "-o",
            c="tab:orange",
            markersize=marker_size,
        )
        axs[1, index].plot(
            locations,
            pca_2[index * num_markers : index * num_markers + num_markers],
            "x-",
            c="tab:blue",
            markersize=marker_size,
        )
        axs[1, index].plot(
            locations,
            rpca_2[index * num_markers : index * num_markers + num_markers],
            "-v",
            c="tab:green",
            markersize=marker_size,
        )
        #     axs[1, index].set_ylabel("Avg. % Relative Error", size = 30)
        axs[1, index].set_xlabel(subfigure_caption, size=text_size, labelpad=axis_pad)

        axs[0, index].set_xticks(ticks=locations, labels=feature_2, size=20)
        axs[1, index].set_xticks(ticks=locations, labels=feature_2, size=20)

        axs[0, index].set_yticks(
            ticks=np.arange(0, max_y1, y1_step),
            labels=np.arange(0, max_y1, y1_step),
            size=text_size,
        )
        axs[1, index].set_yticks(
            ticks=np.arange(0, max_y2, y2_step),
            labels=np.arange(0, max_y2, y2_step),
            size=text_size,
        )

    plt.suptitle(suptitle, fontsize=title_size)

    # Legend
    red_patch = mlines.Line2D(
        [],
        [],
        color="tab:blue",
        marker="x",
        linestyle="-",
        markersize=marker_size,
        label="PCA",
    )
    blue_patch = mlines.Line2D(
        [],
        [],
        color="tab:orange",
        marker="o",
        linestyle="-",
        markersize=marker_size,
        label="EPCA",
    )
    green_patch = mlines.Line2D(
        [],
        [],
        color="tab:green",
        marker="v",
        linestyle="-",
        markersize=marker_size,
        label="RPCA",
    )
    axs[1, len(feature_1) - 1].legend(
        handles=[red_patch, blue_patch, green_patch],
        loc="lower right",
        fontsize=text_size,
    )

    rows = ["PC {}".format(row) for row in ["1", "2"]]
    pad = 5  # in points
    for ax, row in zip(axs[:, 0], rows):
        ax.annotate(
            row,
            xy=(0, 0.5),
            xytext=(-ax.yaxis.labelpad - pad, 0),
            xycoords=ax.yaxis.label,
            textcoords="offset points",
            size=text_size,
            ha="right",
            va="center",
        )

    # plt.subplots_adjust(
    #                     wspace=.4,
    #                     hspace=.2)

    plt.savefig(filepath + ".eps", format="eps", bbox_inches="tight")

    return fig, axs


def relative_error_plot(
    num_trials: int, filename_roots: List[str], filepath: str, **kwargs
):
    """
    Generate and save plot of relative error in principal components for all methods in different noise domains.

    Args:
        num_trials (int) Number of trials for which to plot data.
        filename_roots (List[str]): List of root filenames for the datasets.
        filepath (str): Filepath to save image to. Do not include file extension.
            Extension is automatically .eps.
    Kwargs:
        figsize (Tuple(int)) Figure size.
    """
    corruptions = ["sparse s&p", "uniform white", "normal white", "outliers"]
    methods = [" pca_performance", " epca_performance", " rpca_performance"]

    figsize = kwargs.get("figsize", (24, 16))

    fig = plt.figure(figsize=figsize)
    spacing = 4
    vspace = 4
    gs = GridSpec(spacing * 2 + 3, spacing * 4 + 1)

    for corruption_index, corruption in enumerate(corruptions):
        pc1 = [[], [], []]

        pc2 = [[], [], []]

        for trial_num in range(num_trials):
            filenames = [
                filename_root + str(trial_num) + ".txt"
                for _, filename_root in enumerate(filename_roots)
            ]

            for filename in filenames:
                try:
                    df = pd.read_csv(filename, header=0, sep=";", index_col=False)
                    for j, method in enumerate(methods):
                        performance = (
                            df[df["data"] == corruption][method]
                            .item()
                            .strip("] [")
                            .split(",")
                        )
                        performance = [float(item) for item in performance]
                        if len(performance) >= 2:
                            pc1[j].append(performance[0])
                            pc2[j].append(performance[1])
                        else:
                            pc1[j].append(np.nan)
                            pc2[j].append(np.nan)
                except:
                    continue

        df1 = pd.DataFrame(np.array(pc1).T, columns=["PCA", "EPCA", "RPCA"])
        df2 = pd.DataFrame(np.array(pc2).T, columns=["PCA", "EPCA", "RPCA"])

        if corruption_index == 0:
            ax1 = fig.add_subplot(gs[1 : vspace + 1, 0:spacing])
            ax2 = fig.add_subplot(
                gs[1 : vspace + 1, spacing : spacing * 2]
            )  # ,sharey=ax1)

            axtitle = fig.add_subplot(gs[0, 0 : spacing * 2])
            axtitle.axis("off")
            axtitle.set_title("(a) Sparse Noise", fontsize=30)

        elif corruption_index == 1:
            ax1 = fig.add_subplot(
                gs[
                    1 : vspace + 1 : spacing * 3 + 1 + 1,
                    spacing * 2 + 1 : spacing * 3 + 1,
                ]
            )
            ax2 = fig.add_subplot(gs[1 : vspace + 1, spacing * 3 + 1 :])  # ,sharey=ax1)

            axtitle = fig.add_subplot(gs[0, spacing * 2 + 1 :])
            axtitle.axis("off")
            axtitle.set_title("(b) Uniform White Noise", fontsize=30)

        elif corruption_index == 2:
            ax1 = fig.add_subplot(gs[vspace + 3 : vspace * 2 + 3, 0:spacing])
            ax2 = fig.add_subplot(
                gs[vspace + 3 : vspace * 2 + 3, spacing : spacing * 2]
            )  # ,sharey=ax1)

            axtitle = fig.add_subplot(gs[vspace + 2, 0 : spacing * 2])
            axtitle.axis("off")
            axtitle.set_title("(c) Normal White Noise", fontsize=30)

        elif corruption_index == 3:
            ax1 = fig.add_subplot(
                gs[vspace + 3 : vspace * 2 + 3, spacing * 2 + 1 : spacing * 3 + 1]
            )
            ax2 = fig.add_subplot(
                gs[vspace + 3 : vspace * 2 + 3, spacing * 3 + 1 :]
            )  # ,sharey=ax1)

            axtitle = fig.add_subplot(gs[vspace + 2, spacing * 2 + 1 :])
            axtitle.axis("off")
            axtitle.set_title("(d) Outliers", fontsize=30)

        ax1.set_title("PC 1", size=30)
        ax2.set_title("PC 2", size=30)
        ax1.set_ylabel("% Relative Error", size=30)

        sns.boxplot(data=df1, ax=ax1, showfliers=False)
        sns.boxplot(data=df2, ax=ax2, showfliers=False)

        ax1.tick_params(axis="y", which="major", labelsize=25)
        ax1.tick_params(axis="x", which="major", labelsize=25)
        ax2.tick_params(axis="x", which="major", labelsize=25)
        ax2.tick_params(axis="y", which="major", labelsize=25)

    plt.subplots_adjust(wspace=3, hspace=1)

    plt.savefig(filepath + ".eps", format="eps", bbox_inches="tight")

    plt.show()

    return fig


def runtime_summary_plot(
    num_trials: int,
    filepath: str,
    datasets: List[str],
    root_filenames=List[str],
    **kwargs
):
    """

    Kwargs:
        titlesize (float): Title size.
        textsize (float): Text size.
    """
    corruptions = ["sparse s&p", "uniform white", "normal white", "outliers"]
    runtimes = [" pca_runtime", " epca_runtime", " rpca_runtime"]
    methods = ["PCA", "EPCA", "RPCA"]
    runtime_data = {"Runtime": [], "Method": [], "Dataset": []}

    for corruption in corruptions:
        for trial_num in range(num_trials):
            filenames = [
                root_filename + str(trial_num) + ".txt"
                for root_filename in root_filenames
            ]

            for index, filename in enumerate(filenames):
                try:
                    df = pd.read_csv(filename, header=0, sep=";", index_col=False)
                    for j, method in enumerate(methods):
                        runtime_val = float(df[df["data"] == corruption][runtimes[j]])

                        if not np.isnan(runtime_val):
                            runtime_data["Runtime"].append(runtime_val)
                            runtime_data["Method"].append(method)
                            runtime_data["Dataset"].append(datasets[index])

                        else:
                            runtime_data["Runtime"].append(np.nan)
                            runtime_data["Method"].append(method)
                            runtime_data["Dataset"].append(datasets[index])

                except:
                    continue

    df = pd.DataFrame(data=runtime_data)

    fig, axs = plt.subplots(1, 1, figsize=(20, 15), sharey="col")

    titlesize = kwargs.pop("titlesize", 30)
    textsize = kwargs.pop("textsize", 25)

    sns.boxplot(data=df, x="Dataset", y="Runtime", hue="Method", width=0.80)
    axs.tick_params(axis="x", which="major", labelsize=textsize)

    axs.set_yscale("log")
    axs.tick_params(axis="y", which="major", labelsize=textsize)
    plt.legend(fontsize=textsize)
    plt.title("Runtime Comparison", fontsize=titlesize)
    plt.scatter([5.25], [120], marker="*", s=500, c="g")
    axs.set_xlabel("Dataset", fontsize=textsize)
    axs.set_ylabel("Runtime", fontsize=textsize)

    blue_patch = mpatches.Patch(color="tab:blue", label="PCA")
    orange_patch = mpatches.Patch(color="tab:orange", label="EPCA")
    green_patch = mpatches.Patch(color="tab:green", label="RPCA")
    star = mlines.Line2D(
        [], [], color="k", marker="*", linestyle="None", markersize=20, label="Timeout"
    )
    plt.legend(
        handles=[blue_patch, orange_patch, green_patch, star],
        loc="upper left",
        fontsize=textsize,
    )

    [axs.axvline(x + 0.5, color="grey") for x in axs.get_xticks()]

    plt.savefig(filepath + ".eps", format="eps", bbox_inches="tight")

    plt.show()

    return fig
