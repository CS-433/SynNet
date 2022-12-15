import pandas as pd
import numpy as np
from tdc import Evaluator
from tdc import Oracle
from rdkit.Chem import Descriptors
import rdkit.Chem as Chem

def sanitize(input_file):
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
    n_recovered = len(recovered)
    n_unrecovered = len(unrecovered)
    similarity = unrecovered["similarity"].tolist()

    recovery_rate = n_recovered/n_total * 100

    n_finished = n_recovered + n_unrecovered
    n_unfinished = n_total - n_finished

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



def metrics_for_figure4(recovered):
    """
    Compute various metrics for figure 4

    Args:
        Recovered: Dataframe, The successfully recovered molecules
        Unrecovered: Dataframe, The unsuccessfully recovered molecules

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

    MW = Descriptors.ExactMolWt(Chem.MolFromSmiles('CC'))
    score_target["MW"] = [Descriptors.ExactMolWt(Chem.MolFromSmiles(smi)) for smi in recovered["targets"]]
    score_decoded["MW"] = [Descriptors.ExactMolWt(Chem.MolFromSmiles(smi)) for smi in recovered["decoded"]]

    return score_target, score_decoded