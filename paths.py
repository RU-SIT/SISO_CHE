"""
Portable path defaults for this repository.

Override with environment variables (optional):
  WIRELESS_REPO_ROOT          Project root (default: directory containing this file)
  WIRELESS_DATA_ROOT          Full path to default UMi interpolated dataset (--root for many trainers)
  WIRELESS_DATA_ROOT_TDL      Full path to default TDL interpolated dataset
  WIRELESS_DATA_ROOT_PSPACING Full path to UMi p-spacing dataset (plots / MSE scripts)
  WIRELESS_CHECKPOINTS_ROOT   Where TDL_updated_model, MD_updated_models, etc. live
                              (default: same as repo root)
  WIRELESS_CHANNELNET_KERAS   Keras file for DNCNN fine-tuning in models.py

Place downloaded data under <repo>/Sionna_datasets/... or set the env vars above.
"""
from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    r = os.environ.get("WIRELESS_REPO_ROOT")
    if r:
        return Path(r).expanduser().resolve()
    return Path(__file__).resolve().parent


def checkpoints_root() -> Path:
    c = os.environ.get("WIRELESS_CHECKPOINTS_ROOT")
    if c:
        return Path(c).expanduser().resolve()
    return repo_root()


def default_dataset_umi_interpolated() -> str:
    e = os.environ.get("WIRELESS_DATA_ROOT")
    if e:
        return str(Path(e).expanduser().resolve())
    return str(
        repo_root().joinpath(
            "Sionna_datasets", "ps2_p612", "speed5", "SISO-UMi", "interpolated_noleak"
        )
    )


def default_dataset_tdl_interpolated() -> str:
    e = os.environ.get("WIRELESS_DATA_ROOT_TDL")
    if e:
        return str(Path(e).expanduser().resolve())
    return str(
        repo_root().joinpath(
            "Sionna_datasets", "ps2_p612", "speed5", "SISO-TDL", "interpolated_noleak"
        )
    )


def default_dataset_umi_pspacing() -> str:
    e = os.environ.get("WIRELESS_DATA_ROOT_PSPACING")
    if e:
        return str(Path(e).expanduser().resolve())
    return str(
        repo_root().joinpath(
            "Sionna_datasets",
            "ps2_p612",
            "speed5",
            "SISO_pspacing_4",
            "speed5",
            "interpolated_noleak",
        )
    )


def default_dataset_umi_pspacing_17() -> str:
    return str(
        repo_root().joinpath(
            "Sionna_datasets", "ps2_p612", "speed5", "SISO_pspacing_17", "speed5"
        )
    )


def default_dataset_tdl_siso_folder() -> str:
    """TDL SISO folder without `interpolated_noleak` (some iMAML scripts)."""
    e = os.environ.get("WIRELESS_DATA_ROOT_TDL_SISO")
    if e:
        return str(Path(e).expanduser().resolve())
    return str(
        repo_root().joinpath("Sionna_datasets", "ps2_p612", "speed5", "SISO-TDL")
    )


def default_imaml_inner_step_analysis_dir() -> str:
    return str(repo_root() / "imaml_inner_step_analysis")


def default_save_init_umi() -> str:
    return str(checkpoints_root() / "SISO_UMi_init" / "std_scaler_interpolated_noleak")


def default_dataset_umi_siso_folder() -> str:
    """UMi SISO folder without `interpolated_noleak` (some iMAML scripts use this layout)."""
    e = os.environ.get("WIRELESS_DATA_ROOT_UMI_SISO")
    if e:
        return str(Path(e).expanduser().resolve())
    return str(
        repo_root().joinpath("Sionna_datasets", "ps2_p612", "speed5", "SISO-UMi")
    )


def default_imaml_save_dir() -> str:
    """Top-level save directory for wireless iMAML examples (checkpoints under SISO_UMi_init)."""
    return str(checkpoints_root() / "SISO_UMi_init")


def default_save_init_tdl_pcgrad() -> str:
    return str(checkpoints_root() / "SISO_TDL_init" / "pcgrad_std_scaler_interpolated_noleak")


def default_tdl_updated_model() -> str:
    return str(checkpoints_root() / "TDL_updated_model")


def default_tdl_init_dir() -> str:
    return str(repo_root() / "TDL_init")


def default_md_updated_models_los() -> str:
    return str(checkpoints_root() / "MD_updated_models" / "LOS" / "speed_15")


def default_early_stopping_tdl_maml() -> str:
    return str(
        checkpoints_root()
        / "early_stopping_models_tdl"
        / "SNR_grouping"
        / "meta_model_nway_5"
    )


def default_siso_umi_tiny_maml() -> str:
    return str(checkpoints_root() / "SISO_UMi_init" / "tiny_MAML" / "meta_model_nway_4")


def default_umami_maml_adamw() -> str:
    return str(
        checkpoints_root() / "MD_updated_models" / "LOS" / "speed_15" / "adamW_v2" / "meta_model_nway_5"
    )


def channelnet_dncnn_keras() -> str:
    p = os.environ.get("WIRELESS_CHANNELNET_KERAS")
    if p:
        return str(Path(p).expanduser().resolve())
    return str(repo_root() / "ChannelNet" / "DNCNN_final_model.keras")


def default_cebed_output() -> str:
    return str(repo_root() / "MD_datexperiment_SISO" / "lOS")


def default_prediction_plots_dir() -> str:
    return str(repo_root() / "prediction_plots_comparison")


def default_fourier_analysis_dir() -> str:
    return str(repo_root() / "fourier_analysis_snr_results")


def default_inner_loop_tracking_umi() -> str:
    return str(repo_root() / "inner_loop_tracking_data_umi")


def default_inner_loop_tracking_tdl() -> str:
    return str(repo_root() / "inner_loop_tracking_data_tdl")


def default_umi_task_performance_out() -> str:
    return str(repo_root() / "umi_task_performance_analysis")


def default_tdl_task_performance_out() -> str:
    return str(repo_root() / "tdl_task_performance_analysis")


def default_combined_task_performance_out() -> str:
    return str(repo_root() / "combined_task_performance_analysis")


def model_paths_plot() -> dict:
    """Paths for plot_model_predictions.MODEL_PATHS."""
    cr = checkpoints_root()
    rr = repo_root()
    mg = rr / "multigrade_maml" / "multigrade_maml_results"
    return {
        "TDL": {
            "iMAML": str(cr / "TDL_updated_model" / "meta_model_nway_4"),
            "ChannelNet": str(cr / "TDL_updated_model"),
            "MultigradeMAML": str(mg / "TDL" / "finetuning"),
            "CG_MAML": str(cr / "TDL_updated_model" / "CG_MAML" / "meta_model_nway_4"),
        },
        "UMi": {
            "iMAML": str(
                cr
                / "MD_updated_models"
                / "LOS"
                / "speed_15"
                / "IMAML"
                / "100_innerstep"
                / "meta_model_nway_4"
            ),
            "ChannelNet": str(cr / "MD_updated_models" / "LOS" / "speed_15"),
            "MultigradeMAML": str(mg / "UMi" / "finetuning"),
        },
    }


def model_paths_calculate_mse() -> dict:
    """Paths for calculate_mse.MODEL_PATHS."""
    cr = checkpoints_root()
    rr = repo_root()
    mg = rr / "multigrade_maml" / "multigrade_maml_results"
    return {
        "TDL": {
            "MAML": str(
                cr / "early_stopping_models_tdl" / "SNR_grouping" / "meta_model_nway_5"
            ),
            "iMAML": str(cr / "TDL_updated_model" / "meta_model_nway_4"),
            "ChannelNet": str(cr / "TDL_updated_model"),
            "MultigradeMAML": str(mg / "TDL" / "finetuning"),
        },
        "UMi": {
            "MAML": str(
                cr
                / "MD_updated_models"
                / "LOS"
                / "speed_15"
                / "adamW_v2"
                / "meta_model_nway_5"
            ),
            "iMAML": str(
                cr
                / "MD_updated_models"
                / "LOS"
                / "speed_15"
                / "IMAML"
                / "100_innerstep"
                / "meta_model_nway_4"
            ),
            "ChannelNet": str(cr / "MD_updated_models" / "LOS" / "speed_15"),
            "MultigradeMAML": str(mg / "UMi" / "finetuning"),
            "tiny_MAML": str(
                cr / "SISO_UMi_init" / "tiny_MAML" / "meta_model_nway_4"
            ),
        },
    }


def gt_paths_dict() -> dict:
    return {
        "TDL": str(Path(default_tdl_init_dir()) / "channel_label_dict.npy"),
        "UMi": str(Path(default_dataset_umi_pspacing()) / "channel_label_dict.npy"),
    }


def input_paths_dict() -> dict:
    return {
        "TDL": str(Path(default_tdl_init_dir()) / "channel_data_dict.npy"),
        "UMi": str(Path(default_dataset_umi_pspacing()) / "channel_data_dict.npy"),
    }


def imaml_examples_dir() -> str:
    e = os.environ.get("WIRELESS_IMAML_EXAMPLES_DIR")
    if e:
        return str(Path(e).expanduser().resolve())
    return str(repo_root() / "imaml_dev" / "examples")


def imaml_task_defs_dir() -> str:
    return str(Path(imaml_examples_dir()) / "task_defs")
