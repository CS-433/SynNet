import pandas as pd
import numpy as np
from tdc import Evaluator
from tdc import Oracle
from rdkit.Chem import Descriptors
import rdkit.Chem as Chem
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from anaconda3.Lib.pathlib import Path


def sanitize(input_file: Path):
    """
    Separate recovered (similarity = 1) from unrecovered (similarity in ]0,1[)
    and discarding NaNs (similarity = 0)

    Args:
        input_file: String describing path to dataset

    Returns:
        Recovered: Dataframe, The successfully recovered molecules
        Unrecovered: Dataframe, The unsuccessfully recovered molecules
        n_total: total number of molecules analysed
    """
    # Keep track of successfully and unsuccessfully recovered molecules
    recovered = pd.DataFrame({"targets": [], "decoded": [], "similarity": []})
    unrecovered = pd.DataFrame({"targets": [], "decoded": [], "similarity": []})

    result_df = pd.read_csv(input_file, compression='gzip')

    # Split smiles, discard NaNs
    is_recovered = result_df["similarity"] == 1.0
    unrecovered = pd.concat([unrecovered, result_df[~is_recovered].dropna()], ignore_index=True)
    recovered = pd.concat([recovered, result_df[is_recovered].dropna()], ignore_index=True)

    n_total = len(result_df["decoded"])

    return recovered, unrecovered, n_total


def metrics_for_table1(recovered, unrecovered, n_total):
    """
    Compute various metrics for table 1

    Args:
        Recovered: Dataframe, The successfully recovered molecules
        Unrecovered: Dataframe, The unsuccessfully recovered molecules
        n_total: total number of molecules analysed

    Returns:
        recovery_rate: Percentage of molecules recovered
        average_similarity: Average similarity of molecules
        kl_divergence: Kullback-Leibler (KL) divergence
        fc_distance: Frechet ChemNet Distance (FCD)
    """

    similarity = unrecovered["similarity"].tolist()

    n_recovered = len(recovered)
    recovery_rate = n_recovered/n_total * 100

    average_similarity = np.mean(similarity)

    temp = []
    # Evaluate on TDC evaluators
    for metric in "KL_divergence FCD_Distance".split():
        evaluator = Evaluator(name=metric)
        try:
            score_recovered = evaluator(recovered["targets"], recovered["decoded"])
            score_unrecovered = evaluator(unrecovered["targets"], unrecovered["decoded"])
        except TypeError:
            # Some evaluators only take 1 input args, try that.
            score_recovered = evaluator(recovered["decoded"])
            score_unrecovered = evaluator(unrecovered["decoded"])

        temp.append(score_recovered)
        temp.append(score_unrecovered)

    kl_divergence = temp[0:2]
    fc_distance = temp[2:4]
    return recovery_rate, average_similarity, kl_divergence, fc_distance


def display_table1(input_file: Path):
    """
    Compute various metrics for table 1 and display it in a notebook

    Args:
        input_file: String describing path to dataset
    """
    recovered, unrecovered, n_total = sanitize(input_file)
    recovery_rate, average_similarity, kl_divergence, fc_distance = metrics_for_table1(recovered, unrecovered, n_total)
    df = pd.DataFrame(data={"N": n_total, "Recovery Rate % ": recovery_rate, "Average Similarity ": average_similarity, "KL Divergence ": kl_divergence, "FC Distance ": fc_distance}, index=["Reachable", "Unreachable"])
    display(df)


def metrics_for_figure4(recovered):
    """
    Compute various metrics for figure 4

    Args:
        Recovered: Dataframe, The successfully recovered molecules

    Returns:
        score_target: contains SA, LogP, QED, MW scores for target molecules
        score_decoded: contains SA, LogP, QED, MW scores for target molecules
    """
    score_target = {}
    score_decoded = {}

    for func in "SA LogP QED".split():
        oracle = Oracle(name = func)
        score_target[func] = [oracle(smi) for smi in recovered["targets"]]
        score_decoded[func] = [oracle(smi) for smi in recovered["decoded"]]

    score_target["MW"] = [Descriptors.ExactMolWt(Chem.MolFromSmiles(smi)) for smi in recovered["targets"]]
    score_decoded["MW"] = [Descriptors.ExactMolWt(Chem.MolFromSmiles(smi)) for smi in recovered["decoded"]]

    return score_target, score_decoded

def produce_figure4(input_file: Path, title):
    """
    Compute various metrics for table 1 and display it in a notebook

    Args:
        input_file: Path to dataset
    """
    recovered, unrecovered, n_total = sanitize(input_file)
    score_target, score_decoded = metrics_for_figure4(recovered)

    xlim = [[1,7],[-10,10],[0,1000],[0,1]]
    ylim = [[1,7],[-10,10],[0,1000],[0,1]]
    text_pos = [[1.5,6.5],[-8,8.5],[100,900],[0.1,0.9]]

    fig, axs = plt.subplots(1,4, figsize=(20, 5))
    fig.suptitle(title+" Molecules", fontsize=15)
    for idx, func in enumerate("SA LogP MW QED".split()):
        target = score_target[func]
        decoded = score_decoded[func]

        # compute regression score function
        r2 = r2_score(decoded,target)

        axs[idx].plot(np.linspace(xlim[idx][0], xlim[idx][1], 10), np.linspace(ylim[idx][0], ylim[idx][1], 10), 'r--')
        axs[idx].plot(decoded, target, 'x')
        axs[idx].set_xlabel(func + " Score", fontsize=15)
        axs[idx].text(text_pos[idx][0],text_pos[idx][1], "$r^2$ = "+ '{:.3}'.format(r2), fontsize=15)
        axs[idx].set_xlim(xlim[idx])
        axs[idx].set_ylim(ylim[idx])

    axs[0].set_ylabel("Recovered Value", fontsize=15)
    fig.tight_layout()

    plt.savefig(title + "Figure4.png")

