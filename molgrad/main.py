import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm

from molgrad.net import MPNNPredictor
from molgrad.train import DEVICE
from molgrad.utils import MODELS_PATH
from molgrad.vis import molecule_importance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s || %(name)s | %(levelname)s: %(message)s",
    datefmt="%Y/%m/%d %I:%M:%S %p",
)
LOGGER = logging.getLogger("molgrad")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate structural and global importances for a molecule using a message-passing neural-network approach."
    )
    parser.add_argument(
        "-model_path",
        dest="model_path",
        type=str,
        required=False,
        default=os.path.join(MODELS_PATH, "herg_noHs.pt"),
        help="Path the weights of a trained MPNN model. Defaults to the pretrained models/herg_noHs.pt",
    )

    parser.add_argument(
        "-node_in_feats",
        dest="node_in_feats",
        type=int,
        required=False,
        default=49,
        help="Number of node features considered in the MPNN model. Defaults to 49.",
    )

    parser.add_argument(
        "-edge_in_feats",
        dest="edge_in_feats",
        type=int,
        required=False,
        default=10,
        help="Number of edge features considered in the MPNN model. Defaults to 10.",
    )

    parser.add_argument(
        "-global_feats",
        dest="global_feats",
        type=int,
        required=False,
        default=4,
        help="Number of global features considered in the MPNN model. Defaults to 4.",
    )

    parser.add_argument(
        "-smi",
        dest="smi",
        type=str,
        required=True,
        help="SMILES string or path to a valid .smi file with several SMILES separated by newlines.",
    )

    parser.add_argument(
        "-n_steps",
        dest="n_steps",
        type=int,
        required=False,
        default=50,
        help="Number of steps used in the Riemann approximation of the integral. Defaults to 50.",
    )

    parser.add_argument(
        "-version",
        dest="version",
        type=int,
        required=False,
        default=2,
        help="Version of the implemented algorithm to use. Check molgrad/vis.py for details. Defaults to 2.",
    )

    parser.add_argument(
        "-eps",
        dest="eps",
        type=float,
        required=False,
        default=0.0001,
        help="Minimum gradient value to show. Defaults to 1e-4.",
    )

    parser.add_argument(
        "-vis_factor",
        dest="vis_factor",
        type=float,
        required=False,
        default=1.0,
        help="Multiplicative scalar factor for the gradients.",
    )

    parser.add_argument(
        "-feature_scale",
        dest="feature_scale",
        type=bool,
        required=False,
        default=True,
        help="Scales the gradients by the original features.",
    )

    parser.add_argument(
        "-add_hs",
        dest="add_hs",
        type=bool,
        required=False,
        default=False,
        help="Whether to add hydrogens to the provided molecules",
    )

    parser.add_argument(
        "-img_width",
        dest="img_width",
        type=int,
        required=False,
        default=400,
        help="Width of the resulting .svg canvas",
    )

    parser.add_argument(
        "-img_height",
        dest="img_height",
        type=int,
        required=False,
        default=200,
        help="Height of the resulting .svg canvas",
    )

    parser.add_argument(
        "-output_f",
        dest="output_f",
        type=str,
        required=True,
        help="Output path where to store results",
    )

    args = parser.parse_args()

    if torch.cuda.is_available():
        LOGGER.info(
            f"Using device {torch.cuda.get_device_name()} for prediction and feature attribution."
        )
    else:
        LOGGER.warning(
            f"A CUDA-capable device was not found. Using CPU for prediction and feature attribution, which can take considerably longer."
        )

    LOGGER.info("Loading model...")

    model = MPNNPredictor(
        node_in_feats=args.node_in_feats,
        edge_in_feats=args.edge_in_feats,
        global_feats=args.global_feats,
        n_tasks=1,
    ).to(DEVICE)

    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))

    LOGGER.info(f"Model {args.model_path} successfully loaded!")

    # Procesing ligands

    if args.smi.endswith(".smi"):
        with open(args.smi, "r+") as handle:
            ligands = handle.readlines()
        ligands = [sm.strip("\n") for sm in ligands]
        if ligands[-1] == "":
            ligands.pop(-1)

    else:
        ligands = [args.smi]

    LOGGER.info("Now computing attributions...")

    svgs = []
    g_imp = []
    failed = []

    for idx, sm in tqdm(enumerate(ligands)):
        mol = MolFromSmiles(sm)
        if mol is None:
            LOGGER.error(f"RDKit could read correctly input with SMILES {sm}")
            failed.append(sm)
        else:
            svg, _, _, _, global_importance = molecule_importance(
                mol,
                model,
                n_steps=args.n_steps,
                version=args.version,
                eps=args.eps,
                vis_factor=args.vis_factor,
                feature_scale=args.feature_scale,
                img_width=args.img_width,
                img_height=args.img_height,
                addHs=args.add_hs,
            )
            svgs.append(svg)
            g_imp.append(global_importance)

    g_imp = np.vstack(g_imp)
    df = pd.DataFrame(
        index=np.setdiff1d(np.arange(len(ligands)), failed),
        data=g_imp,
        columns=[f"global_{idx}" for idx in range(args.global_feats)],
    )

    os.makedirs(args.output_f, exist_ok=True)
    os.makedirs(os.path.join(args.output_f, "svg"), exist_ok=True)
    df.to_csv(os.path.join(args.output_f, "global.csv"), index_label="idx")

    for idx, svg in zip(df.index, svgs):
        with open(os.path.join(args.output_f, "svg", f"{idx}.svg"), "w+") as handle:
            handle.write(svg)

