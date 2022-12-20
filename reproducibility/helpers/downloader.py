import re
import random

from .file_utils import *
from .paths import (
    data_folder,
    intermediates_folder,
    download_folder,
    molecules_folder,
    test_set_path,
)

import requests
from tqdm.auto import tqdm
import pandas as pd
from rdkit import Chem

from synnet.data_generation.preprocessing import BuildingBlockFileHandler
from synnet.models.mlp import MLP
from synnet.utils.data_utils import SyntheticTreeSet

intermediates_folder.mkdir(exist_ok=True)
download_folder.mkdir(exist_ok=True)
molecules_folder.mkdir(exist_ok=True)


def download_file(
    url: str, output: Path, force: bool, progress_bar: bool = True
) -> Path:
    """
    Download a file and store it in the provided output path.

    If the file already exist, it will not be downloaded. This can be bypassed with the force flag.

    A progress bar will be displayed, it can be disabled.

    Args:
        url: The url of the file to download
        output: Where to store the downloaded file, can be a directory
        force: If True and the file already exists, it will be re-downloaded
        progress_bar (True): If True, a progress bar of the download will be displayed

    Returns:
        The Path to the downloaded file
    """
    # Make an HTTP request within a context manager
    with requests.get(url, stream=True) as r:
        # Retrieve file name and update output if needed
        if output.is_dir():
            if "Content-Disposition" in r.headers.keys():
                content = r.headers["Content-Disposition"]
                name = re.findall('filename="(.+)"', content)[0]
            else:
                name = file_name(url)

            output /= name
        else:
            name = output.name

        if should_skip(name, "download", output, force):
            return output

        # check header to get content length, in bytes
        total_length = int(r.headers.get("Content-Length"))

        stream = r.raw
        if progress_bar:
            # implement progress bar via tqdm
            stream = tqdm.wrapattr(
                r.raw, "read", total=total_length, desc=f"Downloading {name}"
            )

        with stream as raw:
            # save the output to a file
            with output.open("wb") as out:
                shutil.copyfileobj(raw, out)

        return output


def convert_sdf_to_smiles(input_path: Path, sample: int = None) -> list[smile]:
    """
    Convert an input sdf file to a smiles list

    Args:
        input_path: The path to the sdf file
        sample: (Optional) Number of smiles to extract from the sdf file.
                If it is not set, every smiles will be extracted

    Returns:
        A list containing the smiles
    """
    print("Loading sdf...  ", end="")
    supplier = Chem.SDMolSupplier(str(input_path))
    size = len(supplier)
    r = range(size)
    if sample is not None:
        size = sample
        r = random.sample(r, sample)
    print("done")

    # create mol generator and setup progress bar using tqdm
    mols = (
        supplier[i]
        for i in tqdm(r, total=size, desc="Converting sdf to smile", unit="mol")
    )
    return [Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False) for mol in mols]


def get_original_checkpoints(force=False) -> list[MLP]:
    """
    Retrieve the original checkpoint of the model trained by the paper's Authors

    Download and process it if needed. If the file is already present, it will not be recomputed.
    But you can set force to True to bypass any existing intermediate file.

    Args:
        force (False): If set, any stored file will be bypassed

    Returns:
        A list containing the checkpoints
    """

    tar_file = download_folder / "original_checkpoints.tar.gz"
    checkpoint_path = data_folder / "original_checkpoints"

    # Check that it does not exist yet
    if should_skip("original checkpoints", "compute", checkpoint_path, force):
        return load_checkpoints(checkpoint_path)

    # retrieve the checkpoint tar file
    download_file("https://figshare.com/ndownloader/files/31067692", tar_file, force)
    # extract it
    extract_output = extract_tar(tar_file, intermediates_folder, force)
    checkpoint_temp_path = extract_output / "hb_fp_2_4096_256"

    # move original_checkpoints to their respective folders
    checkpoint_path.mkdir()

    for model in ["act", "rt1", "rxn", "rt2"]:
        model_path = checkpoint_path / model
        model_path.mkdir()

        shutil.move(
            checkpoint_temp_path / f"{model}.ckpt",
            model_path / "ckpts.dummy-val_loss=0.00.ckpt",
        )

    return load_checkpoints(checkpoint_path)


def get_trained_checkpoints(force=False) -> list[MLP]:
    """
    Retrieve the checkpoint of the model we trained

    Download and process it if needed. If the file is already present, it will not be recomputed.
    But you can set force to True to bypass any existing intermediate file.

    Args:
        force (False): If set, any stored file will be bypassed

    Returns:
        A list containing the checkpoints
    """

    checkpoint_path = data_folder / "trained_checkpoints"

    models = {
        "act": "https://drive.switch.ch/index.php/s/w8uo1CmI6JZKoa1/download",
        "rt1": "https://drive.switch.ch/index.php/s/3MPFuEkW4Y2BPj5/download",
        "rxn": "https://drive.switch.ch/index.php/s/gZIiEIufxC2LYUp/download",
        "rt2": "https://drive.switch.ch/index.php/s/N2hmZiJDBJpLYWi/download",
    }

    # Check that it does not exist yet
    if should_skip("trained checkpoints", "compute", checkpoint_path, force):
        return load_checkpoints(checkpoint_path)

    checkpoint_path.mkdir()
    for model, url in models.items():
        model_tar = download_file(url, download_folder, force)
        extracted_path = extract_tar(model_tar, checkpoint_path, force)
        extracted_path.rename(checkpoint_path / model)

    return load_checkpoints(checkpoint_path)


def get_building_blocks(force=False) -> list[smile]:
    """
    Retrieve the building blocks smiles.

    Download and process it if needed. If the file is already present, it will not be recomputed.
    But you can set force to True to bypass any existing intermediate file.

    Args:
        force (False): If set, any stored file will be bypassed

    Returns:
        A list containing the building block smiles
    """

    output_path = (
        data_folder / "assets" / "building-blocks" / "enamine-us-smiles.csv.gz"
    )

    if should_skip("building_blocks", "compute", output_path, force):
        smiles = BuildingBlockFileHandler().load(str(output_path))
        return smiles

    # download building_block sdf
    sdf_file = download_file(
        "https://drive.switch.ch/index.php/s/zLDApVjC7bU5qx2/download",
        download_folder,
        force,
    )
    # convert sdf to smiles
    smiles = convert_sdf_to_smiles(sdf_file)
    # save it
    BuildingBlockFileHandler().save(str(output_path), smiles)

    return smiles


def get_test_set() -> list[smile]:
    print("Loading test set...")
    return load_smiles(test_set_path)


def get_chembl_dataset(sample_size: int = None, force=False) -> list[smile]:
    """
    Retrieve the ChEMBL smiles.

    Website: https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/

    Download and process it if needed. If the file is already present, it will not be recomputed.
    But you can set force to True to bypass any existing intermediate file.

    Args:
        sample_size (Optional): If set, only a sample of the dataset will be retrieved
        force: If set, any stored file will be bypassed

    Returns:
        A list containing the ChEMBL smiles
    """

    sdf_file = intermediates_folder / "chembl_31.sdf"
    output_path = molecules_folder / "chembl-smiles.csv.gz"

    # Check that it does not exist yet
    if should_skip("ChEMBL", "compute", output_path, force):
        # Load the data
        smiles = load_smiles(output_path)
        if sample_size and len(smiles) != sample_size:
            print(
                "The number of samples differs from the expected amount. The dataset will be recomputed"
            )
            safe_remove(output_path)
        else:
            return smiles

    # Download and extract ChEMBL sdf
    compressed_sdf = download_file(
        "https://drive.switch.ch/index.php/s/jXuJyFIbADdSJkR/download",
        download_folder,
        force,
    )
    decompress_file(compressed_sdf, sdf_file, force)

    # Convert sdf to smiles
    smiles = convert_sdf_to_smiles(sdf_file, sample=sample_size)
    # save it
    save_smiles(smiles, output_path)

    return smiles


def get_zinc_dataset(sample_size=None, force=False) -> list[str]:
    """
    Retrieve the ZINC smiles.

    Website: https://zinc.docking.org/tranches/home/

    Download and process it if needed. If the file is already present, it will not be recomputed.
    But you can set force to True to bypass any existing intermediate file.

    Args:
        sample_size (Optional): If set, only a sample of the dataset will be retrieved
        force: If set, any stored file will be bypassed

    Returns:
        A list containing the ZINC smiles
    """

    urls_path = data_folder / "ZINC-downloader-2D.csv"
    output_path = molecules_folder / "zinc-smiles.csv.gz"

    # Check whether this process should be skipped
    if should_skip("ZINC", "compute", output_path, force):
        # Load the data
        smiles = load_smiles(output_path)
        if sample_size and len(smiles) != sample_size:
            print(
                "The number of samples differs from the expected amount. The dataset will be recomputed"
            )
            safe_remove(output_path)
        else:
            return smiles

    # Download the dataset
    links = pd.read_csv(urls_path, header=None)

    if sample_size is not None:
        # Shuffle the link list such that we take samples from randomly chosen links
        # Use a seed generated by python's random, that way it depends on this random seed
        #
        # This is not perfect as the correlation between molecules in the same file is high.
        # But it avoids downloading the whole dataset
        links = links.sample(frac=1, random_state=random.randint(0, 2**32 - 1))

    # Retrieve elements from the column
    links = links[0]

    master_df = pd.DataFrame()

    pbar = tqdm(total=sample_size)  # Create a manual progress bar

    zinc_downloads = download_folder / "zinc"
    zinc_downloads.mkdir(exist_ok=True)

    for link in links:
        name = file_name(link)
        # Update progress bar
        pbar.set_description(f"Downloading {name}")
        pbar.update(master_df.shape[0])

        path = zinc_downloads / name

        # Retrieve smiles from the link
        download_file(link, path, force, progress_bar=False)
        df = pd.read_csv(path, sep="\t")

        master_df = pd.concat((master_df, df), axis=0)
        # Stop if we have enough molecules
        if master_df.shape[0] > sample_size:
            pbar.update(sample_size)
            break

    # Get only the smiles, and exactly the number we want
    smiles = master_df["smiles"].sample(sample_size)
    # Write it into a file
    save_smiles(smiles, output_path)

    return smiles


def get_filtered_syntrees(force=False) -> dict[str, list[dict]]:
    """
    Download the filtered synthetic trees (train, validation and test sets)

    Returns:
        A dict mapping the synthetic trees names to their values
    """

    trees = [
        ("train", "https://drive.switch.ch/index.php/s/sH2FTC0Xna3oDbD/download"),
        ("valid", "https://drive.switch.ch/index.php/s/0F9GX5HR4pgW8KG/download"),
        ("test", "https://drive.switch.ch/index.php/s/dX6c283Ga4OU3kS/download"),
    ]

    output = {}

    for name, url in trees:
        syntree_path = intermediates_folder / f"{name}.json.gz"
        # Download compressed file
        download_file(url, syntree_path, force)
        # Load tree
        output[name] = [
            syntree.output_dict()
            for syntree in SyntheticTreeSet().load(str(syntree_path)).sts
        ]

    return output
