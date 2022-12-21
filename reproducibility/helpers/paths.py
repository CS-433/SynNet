from pathlib import Path

# Where the data files are stored
data_folder = Path("data")
# Where the result files are stored
results_folder = Path("results")

# Where the intermediates file of the downloader are store
intermediates_folder = data_folder / "intermediates"
# Where the downloaded files are stored
download_folder = intermediates_folder / "downloads"

# Where we store the different datasets
molecules_folder = data_folder / "assets" / "molecules"
# Path to the reachable molecules dataset
reachable_smiles_path = molecules_folder / "reachable.csv.gz"


def synthesis_result_path(dataset: str, checkpoint: str) -> Path:
    return results_folder / "synthesis" / dataset / checkpoint


def synthesis_result_csv(dataset: str, checkpoint: str) -> Path:
    return synthesis_result_path(dataset, checkpoint) / "decoded_results.csv.gz"


def optimize_result_path(dataset: str, checkpoint: str) -> Path:
    return results_folder / "optimize" / dataset / checkpoint
