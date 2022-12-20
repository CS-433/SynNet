from pathlib import Path

# Where the data files are stored
data_folder = Path("data")
# Where the result files are stored
results_folder = Path("results")

# Where the intermediates file of the downloader are store
intermediates_folder = Path("intermediates")
# Where the downloaded files are stored
download_folder = intermediates_folder / "downloads"


molecules_folder = data_folder / "assets" / "molecules"

test_set_path = molecules_folder / "test_set.csv.gz"


def synthesis_result_path(dataset: str, checkpoint: str) -> Path:
    return results_folder / "synthesis" / dataset / checkpoint


def optimize_result_path(dataset: str, checkpoint: str) -> Path:
    return results_folder / "optimize" / dataset / checkpoint
