"""Reactant2 network (for predicting 2nd reactant).
"""
import json
import logging
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from synnet.encoding.distances import cosine_distance
from synnet.models.common import get_args, xy_to_dataloader
from synnet.models.mlp import MLP
from synnet.MolEmbedder import MolEmbedder

logger = logging.getLogger(__name__)
MODEL_ID = Path(__file__).stem


def _fetch_molembedder():
    file = args.mol_embedder_file
    logger.info(f"Try to load precomputed MolEmbedder from {file}.")
    molembedder = MolEmbedder().load_precomputed(file).init_balltree(metric=cosine_distance)
    logger.info(f"Loaded MolEmbedder from {file}.")
    return molembedder


if __name__ == "__main__":
    logger.info("Start.")

    # Parse input args
    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")

    pl.seed_everything(0)

    # Set up dataloaders
    dataset = "train"
    train_dataloader = xy_to_dataloader(
        X_file=Path(args.data_dir) / f"X_{MODEL_ID}_{dataset}.npz",
        y_file=Path(args.data_dir) / f"y_{MODEL_ID}_{dataset}.npz",
        n=None if not args.debug else 128,
        batch_size=args.batch_size,
        num_workers=args.ncpu,
        shuffle=True if dataset == "train" else False,
    )

    dataset = "valid"
    valid_dataloader = xy_to_dataloader(
        X_file=Path(args.data_dir) / f"X_{MODEL_ID}_{dataset}.npz",
        y_file=Path(args.data_dir) / f"y_{MODEL_ID}_{dataset}.npz",
        n=None if not args.debug else 128,
        batch_size=args.batch_size,
        num_workers=args.ncpu,
        shuffle=True if dataset == "train" else False,
    )

    logger.info(f"Set up dataloaders.")

    # Fetch Molembedder and init BallTree
    molembedder = None  # _fetch_molembedder()

    INPUT_DIMS = {
        "fp": {
            "hb": int(4 * args.nbits + 91),
            "gin": int(4 * args.nbits + 4700),
        },
        "gin": {
            "hb": int(3 * args.nbits + args.out_dim + 91),
            "gin": int(3 * args.nbits + args.out_dim + 4700),
        },
    }  # somewhat constant...
    input_dims = INPUT_DIMS[args.featurize][args.rxn_template]

    mlp = MLP(
        input_dim=input_dims,
        output_dim=args.out_dim,
        hidden_dim=3000,
        num_layers=5,
        dropout=0.5,
        num_dropout_layers=1,
        task="regression",
        loss="mse",
        valid_loss="mse",
        optimizer="adam",
        learning_rate=3e-4,
        val_freq=10,
        molembedder=molembedder,
        ncpu=args.ncpu,
    )

    # Set up Trainer
    save_dir = Path("results/logs/") / MODEL_ID
    save_dir.mkdir(exist_ok=True, parents=True)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir, name="")
    csv_logger = pl_loggers.CSVLogger(tb_logger.log_dir, name="", version="")
    logger.info(f"Log dir set to: {tb_logger.log_dir}")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=tb_logger.log_dir,
        filename="ckpts.{epoch}-{val_loss:.2f}",
        save_weights_only=False,
    )
    earlystop_callback = EarlyStopping(monitor="val_loss", patience=3)
    tqdm_callback = TQDMProgressBar(refresh_rate=int(len(train_dataloader) * 0.05))

    max_epochs = args.epoch if not args.debug else 100
    # Create trainer
    trainer = pl.Trainer(
        gpus=[0],
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, tqdm_callback],
        logger=[tb_logger, csv_logger],
        fast_dev_run=args.fast_dev_run,
    )

    logger.info(f"Start training")
    trainer.fit(mlp, train_dataloader, valid_dataloader)
    logger.info(f"Training completed.")
