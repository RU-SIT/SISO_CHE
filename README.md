# Few shot Channel Estimation 

This repository is the paper "Characterizing Failures of Deep-Learning Models Under Data Paucity for Wireless Time-Varying Channel Estimation" implementation.

## Data and paths

Download the data [here](https://drive.google.com/drive/folders/10hIn854d219OhhC7hLnUOUD_6revp3U1?usp=sharing), then place it under this repository (for example `Sionna_datasets/...` at the project root) **or** point the scripts to your folders using environment variables.

This repository is meant to hold **source code only**. Generated plots (`.png`, etc.), result tables (`.csv`), NumPy dumps (`.npy`, `.npz`), and model checkpoints are listed in `.gitignore` and are not pushed to GitHub.

Portable defaults are defined in `paths.py`. Common overrides:

| Variable | Meaning |
|----------|---------|
| `WIRELESS_REPO_ROOT` | Project root (default: folder containing `paths.py`) |
| `WIRELESS_DATA_ROOT` | Full path to the default UMi interpolated dataset (used as `--root` in many trainers) |
| `WIRELESS_DATA_ROOT_TDL` | Full path to the default TDL interpolated dataset |
| `WIRELESS_DATA_ROOT_PSPACING` | Full path to the UMi p-spacing dataset (for plotting / MSE scripts) |
| `WIRELESS_CHECKPOINTS_ROOT` | Where `TDL_updated_model`, `MD_updated_models`, `SISO_UMi_init`, etc. live (default: project root) |
| `WIRELESS_CHANNELNET_KERAS` | Keras weights file for `models.fine_tune_DNCNN` (default: `ChannelNet/DNCNN_final_model.keras`) |
| `WIRELESS_DATA_ROOT_UMI_SISO` | UMi `SISO-UMi` folder (without `interpolated_noleak`) for some iMAML scripts |
| `WIRELESS_DATA_ROOT_TDL_SISO` | TDL `SISO-TDL` folder for iMAML analysis scripts |
| `WIRELESS_IMAML_EXAMPLES_DIR` | Override for `imaml_dev/examples` (Omniglot task defs, etc.) |
| `MULTIMNIST_ROOT` | PCGrad demo data in `Pytorch-PCGrad/data/multi_mnist.py` (default: `<repo>/data/MultiMNIST`) |

## Steps

1. Run `channel2D.py` to create a dictionary of data.

2. To train the meta model, run `MAML_trainer.py` (set arguments in the script or on the command line).

3. Run `MAML_finetuning.py` to fine-tune the model and save the fine-tuned model and evaluation.

4. To compare results, run `test.py`.

