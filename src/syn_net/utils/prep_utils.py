"""
This file contains various utils for data preparation and preprocessing.
"""
from typing import Iterator, Union, Tuple
import numpy as np
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from syn_net.utils.data_utils import Reaction, SyntheticTree
from syn_net.utils.predict_utils import (can_react, get_action_mask,
                                         get_reaction_mask, mol_fp,
                                         )

from pathlib import Path
from rdkit import Chem
import logging
logger = logging.getLogger(__name__)

def rdkit2d_embedding(smi):
    """
    Computes an embedding using RDKit 2D descriptors.

    Args:
        smi (str): SMILES string.

    Returns:
        np.ndarray: A molecular embedding corresponding to the input molecule.
    """
    from tdc.chem_utils import MolConvert
    if smi is None:
        return np.zeros(200).reshape((-1, ))
    else:
        # define the RDKit 2D descriptor
        rdkit2d = MolConvert(src = 'SMILES', dst = 'RDKit2D')
        return rdkit2d(smi).reshape(-1, )

import functools
@functools.lru_cache(maxsize=1)
def _fetch_gin_pretrained_model(model_name: str):
    from dgllife.model import load_pretrained
    """Get a GIN pretrained model to use for creating molecular embeddings"""
    device     = 'cpu'
    model      = load_pretrained(model_name).to(device)
    model.eval()
    return model


def organize(st: SyntheticTree, d_mol: int=300, target_embedding: str='fp', radius: int=2, nBits:int=4096,
             output_embedding: str ='gin') -> Tuple(sparse.csc_matrix,sparse.csc_matrix):
    """
    Organizes synthetic trees into states and node states at each step into sparse matrices.

    Args:
        st: Synthetic tree to organize
        d_mol: The molecular embedding size. Defaults to 300
        target_embedding: Embedding for the input node states.
        radius: (if Morgan fingerprint) radius
        nBits: (if Morgan fingerprint) bits
        output_embedding: Embedding for the output node states

    Raises:
        ValueError: Raised if target embedding not supported.

    Returns:
        sparse.csc_matrix: Node states pulled from the tree.
        sparse.csc_matrix: Actions pulled from the tree.
    """


    states = []
    steps = []

    OUTPUT_EMBEDDINGS_DIMS = {
        "gin": 300,
        "fp_4096": 4096,
        "fp_256": 256,
        "rdkit2d": 200,
    }

    d_mol = OUTPUT_EMBEDDINGS_DIMS[output_embedding]

    # Do we need a gin embedder?
    if output_embedding == "gin" or target_embedding == "gin":
        model = _fetch_gin_pretrained_model("gin_supervised_contextpred")

    # Compute embedding of target molecule, i.e. the root of the synthetic tree
    if target_embedding == 'fp':
        target = mol_fp(st.root.smiles, radius, nBits).tolist()
    elif target_embedding == 'gin':
        from syn_net.encoding.gins import get_mol_embedding
        # define model to use for molecular embedding
        target = get_mol_embedding(st.root.smiles, model=model).tolist()
    else:
        raise ValueError('Target embedding only supports fp and gin.')

    most_recent_mol = None
    other_root_mol  = None
    for i, action in enumerate(st.actions):

        most_recent_mol_embedding = mol_fp(most_recent_mol, radius, nBits).tolist()
        other_root_mol_embedding  = mol_fp(other_root_mol, radius, nBits).tolist()
        state = most_recent_mol_embedding + other_root_mol_embedding + target # (3d,1)

        if action == 3:
            step = [3] + [0]*d_mol + [-1] + [0]*d_mol + [0]*nBits

        else:
            r = st.reactions[i]
            mol1 = r.child[0]
            if len(r.child) == 2:
                mol2 = r.child[1]
            else:
                mol2 = None

            if output_embedding == 'gin':
                step = ([action]
                        + get_mol_embedding(mol1, model=model).tolist()
                        + [r.rxn_id]
                        + get_mol_embedding(mol2, model=model).tolist()
                        + mol_fp(mol1, radius, nBits).tolist())
            elif output_embedding == 'fp_4096':
                step = ([action]
                        + mol_fp(mol1, 2, 4096).tolist()
                        + [r.rxn_id]
                        + mol_fp(mol2, 2, 4096).tolist()
                        + mol_fp(mol1, radius, nBits).tolist())
            elif output_embedding == 'fp_256':
                step = ([action]
                        + mol_fp(mol1, 2, 256).tolist()
                        + [r.rxn_id]
                        + mol_fp(mol2, 2, 256).tolist()
                        + mol_fp(mol1, radius, nBits).tolist())
            elif output_embedding == 'rdkit2d':
                step = ([action]
                        + rdkit2d_embedding(mol1).tolist()
                        + [r.rxn_id]
                        + rdkit2d_embedding(mol2).tolist()
                        + mol_fp(mol1, radius, nBits).tolist())

        if action == 2:
            most_recent_mol = r.parent
            other_root_mol = None

        elif action == 1:
            most_recent_mol = r.parent

        elif action == 0:
            other_root_mol = most_recent_mol
            most_recent_mol = r.parent

        states.append(state)
        steps.append(step)

    return sparse.csc_matrix(np.array(states)), sparse.csc_matrix(np.array(steps))

def synthetic_tree_generator(
    building_blocks: list[str], reaction_templates: list[Reaction], max_step: int = 15
) -> tuple[SyntheticTree, int]:
    """
    Generates a synthetic tree from the available building blocks and reaction
    templates. Used in preparing the training/validation/testing data.

    Args:
        building_blocks (list): Contains SMILES strings for purchasable building
            blocks.
        reaction_templates (list): Contains `Reaction` objects.
        max_step (int, optional): Indicates the maximum number of reaction steps
            to use for building the synthetic tree data. Defaults to 15.

    Returns:
        tree (SyntheticTree): The built up synthetic tree.
        action (int): Index corresponding to a specific action.
    """
    # Initialization
    tree = SyntheticTree()
    mol_recent = None
    building_blocks = np.asarray(building_blocks)

    try:
        for i in range(max_step):
            # Encode current state
            state = tree.get_state()

            # Predict action type, masked selection
            # Action: (Add: 0, Expand: 1, Merge: 2, End: 3)
            action_proba = np.random.rand(4)
            action_mask = get_action_mask(tree.get_state(), reaction_templates)
            action = np.argmax(action_proba * action_mask)

            # Select first molecule
            if action == 3: # End
                break
            elif action == 0: # Add
                mol1 = np.random.choice(building_blocks)
            else: # Expand or Merge
                mol1 = mol_recent

            # Select reaction
            reaction_proba = np.random.rand(len(reaction_templates))

            if action != 2: # = action == 0 or action == 1
                rxn_mask, available = get_reaction_mask(smi=mol1,
                                                        rxns=reaction_templates)
            else: # merge tree
                _, rxn_mask = can_react(tree.get_state(), reaction_templates)
                available = [[] for rxn in reaction_templates]

            if rxn_mask is None:
                if len(state) == 1:
                    action = 3
                    break
                else:
                    break

            rxn_id = np.argmax(reaction_proba * rxn_mask)
            rxn = reaction_templates[rxn_id]

            # Select second molecule
            if rxn.num_reactant == 2:
                if action == 2: # Merge
                    temp = set(state) - set([mol1])
                    mol2 = temp.pop()
                else: # Add or Expand
                    mol2 = np.random.choice(available[rxn_id])
            else:
                mol2 = None

            # Run reaction
            mol_product = rxn.run_reaction([mol1, mol2])

            # Update
            tree.update(action, int(rxn_id), mol1, mol2, mol_product)
            mol_recent = mol_product

    except Exception as e:
        print(e)
        action = -1
        tree = None

    if action != 3:
        tree = None
    else:
        tree.update(action, None, None, None, None)

    return tree, action

def prep_data(main_dir, num_rxn, out_dim, datasets=None):
    """
    Loads the states and steps from preprocessed *.npz files and saves data
    specific to the Action, Reactant 1, Reaction, and Reactant 2 networks in
    their own *.npz files.

    Args:
        main_dir (str): The path to the directory containing the *.npz files.
        num_rxn (int): Number of reactions in the dataset.
        out_dim (int): Size of the output feature vectors.
    """
    if datasets is None:
        datasets = ['train', 'valid', 'test']
    main_dir = Path(main_dir)

    for dataset in datasets:

        print(f'Reading {dataset} data ...')
        states_list = []
        steps_list = []

        states_list.append(sparse.load_npz(main_dir / f'states_{dataset}.npz'))
        steps_list.append(sparse.load_npz(main_dir / f'steps_{dataset}.npz'))

        states = sparse.csc_matrix(sparse.vstack(states_list))
        steps = sparse.csc_matrix(sparse.vstack(steps_list))

        # extract Action data
        X = states
        y = steps[:, 0]
        sparse.save_npz(main_dir / f'X_act_{dataset}.npz', X)
        sparse.save_npz(main_dir / f'y_act_{dataset}.npz', y)

        states = sparse.csc_matrix(states.A[(steps[:, 0].A != 3).reshape(-1, )])
        steps = sparse.csc_matrix(steps.A[(steps[:, 0].A != 3).reshape(-1, )])
        print(f'  saved data for "Action"')

        # extract Reaction data
        X = sparse.hstack([states, steps[:, (2 * out_dim + 2):]])
        y = steps[:, out_dim + 1]
        sparse.save_npz(main_dir / f'X_rxn_{dataset}.npz', X)
        sparse.save_npz(main_dir / f'y_rxn_{dataset}.npz', y)
        print(f'  saved data for "Reaction"')

        states = sparse.csc_matrix(states.A[(steps[:, 0].A != 2).reshape(-1, )])
        steps = sparse.csc_matrix(steps.A[(steps[:, 0].A != 2).reshape(-1, )])

        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit([[i] for i in range(num_rxn)])
        # import ipdb; ipdb.set_trace(context=9)

        # extract Reactant 2 data
        X = sparse.hstack(
            [states,
             steps[:, (2 * out_dim + 2):],
             sparse.csc_matrix(enc.transform(steps[:, out_dim+1].A.reshape((-1, 1))).toarray())]
        )
        y = steps[:, (out_dim+2): (2 * out_dim + 2)]
        sparse.save_npz(main_dir / f'X_rt2_{dataset}.npz', X)
        sparse.save_npz(main_dir / f'y_rt2_{dataset}.npz', y)
        print(f'  saved data for "Reactant 2"')

        states = sparse.csc_matrix(states.A[(steps[:, 0].A != 1).reshape(-1, )])
        steps = sparse.csc_matrix(steps.A[(steps[:, 0].A != 1).reshape(-1, )])

        # extract Reactant 1 data
        X = states
        y = steps[:, 1: (out_dim+1)]
        sparse.save_npz(main_dir / f'X_rt1_{dataset}.npz', X)
        sparse.save_npz(main_dir / f'y_rt1_{dataset}.npz', y)
        print(f'  saved data for "Reactant 1"')

    return None

class Sdf2SmilesExtractor:
    """Helper class for data generation."""

    def __init__(self) -> None:
        self.smiles: Iterator[str]

    def from_sdf(self, file: Union[str, Path]):
        """Extract chemicals as SMILES from `*.sdf` file.

        See also:
            https://www.rdkit.org/docs/GettingStartedInPython.html#reading-sets-of-molecules
        """
        file = str(Path(file).resolve())
        suppl = Chem.SDMolSupplier(file)
        self.smiles = (Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False) for mol in suppl)
        logger.info(f"Read data from {file}")

        return self

    def _to_csv_gz(self, file: Path) -> None:
        import gzip

        with gzip.open(file, "wt") as f:
            f.writelines("SMILES\n")
            f.writelines((s + "\n" for s in self.smiles))

    def _to_csv_gz(self, file: Path) -> None:
        with open(file, "wt") as f:
            f.writelines("SMILES\n")
            f.writelines((s + "\n" for s in self.smiles))

    def to_file(self, file: Union[str, Path]) -> None:

        if Path(file).suffixes == [".csv", ".gz"]:
            self._to_csv_gz(file)
        else:
            self._to_txt(file)
        logger.info(f"Saved data to {file}")