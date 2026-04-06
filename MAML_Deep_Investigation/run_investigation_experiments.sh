#!/bin/bash
# MAML Investigation: Experimental Scripts
# This script contains commands to reproduce experiments from the investigation

echo "================================================================"
echo "MAML DEEP INVESTIGATION - EXPERIMENTAL SCRIPTS"
echo "================================================================"
echo ""
echo "This file contains commands to reproduce key experiments."
echo "Uncomment and run the experiments you want to try."
echo ""

# Set base directory (parent of MAML_Deep_Investigation)
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INV_DIR="$BASE_DIR/MAML_Deep_Investigation"

# Create experiment directories
mkdir -p "$INV_DIR/experiments/baseline"
mkdir -p "$INV_DIR/experiments/ablation"
mkdir -p "$INV_DIR/experiments/tracking"

# ================================================================
# EXPERIMENT 1: BASELINE TRAINING (Default Hyperparameters)
# ================================================================
# This reproduces the baseline MAML training with optimal settings
# Expected: Final loss ~0.020, Training time ~4 hours

# python $BASE_DIR/MAML_trainer.py \
#     --root "$BASE_DIR/Sionna_datasets/ps2_p612/speed5/SISO-UMi/interpolated_noleak" \
#     --device "cuda:0" \
#     --save_init "$INV_DIR/experiments/baseline" \
#     --epoch 5000 \
#     --n_way 4 \
#     --k_spt 5 \
#     --k_qry 5 \
#     --batchsz 8 \
#     --meta_lr 5e-4 \
#     --update_lr 1e-3 \
#     --update_step 4 \
#     --max_grad_norm 0.5 \
#     --scheduler_factor 0.5 \
#     --scheduler_patience 8

echo "EXPERIMENT 1: Baseline training configured"
echo "Uncomment lines to run"
echo ""

# ================================================================
# EXPERIMENT 2: TRAINING WITH INNER LOOP TRACKING
# ================================================================
# This tracks inner loop losses for each channel across training
# Useful for analyzing adaptation dynamics

# python $BASE_DIR/MAML_trainer_with_tracking.py \
#     --root "$BASE_DIR/Sionna_datasets/ps2_p612/speed5/SISO-UMi/interpolated_noleak" \
#     --device "cuda:0" \
#     --save_init "$INV_DIR/experiments/tracking" \
#     --tracking_dir "$INV_DIR/experiments/tracking/inner_loop_data" \
#     --epoch 1000 \
#     --n_way 5 \
#     --k_spt 5 \
#     --k_qry 5 \
#     --batchsz 8 \
#     --meta_lr 1e-4 \
#     --update_lr 1e-3 \
#     --update_step 3

echo "EXPERIMENT 2: Inner loop tracking configured"
echo ""

# ================================================================
# EXPERIMENT 3: ABLATION - NO GRADIENT CLIPPING
# ================================================================
# This shows what happens without gradient clipping
# Expected: Training will diverge (NaN loss)

# python $BASE_DIR/MAML_trainer.py \
#     --root "$BASE_DIR/Sionna_datasets/ps2_p612/speed5/SISO-UMi/interpolated_noleak" \
#     --device "cuda:0" \
#     --save_init "$INV_DIR/experiments/ablation/no_clipping" \
#     --epoch 500 \
#     --n_way 4 \
#     --k_spt 5 \
#     --k_qry 5 \
#     --batchsz 8 \
#     --meta_lr 5e-4 \
#     --update_lr 1e-3 \
#     --update_step 4 \
#     --max_grad_norm -1.0

echo "EXPERIMENT 3: No gradient clipping (will diverge!)"
echo ""

# ================================================================
# EXPERIMENT 4: ABLATION - DIFFERENT INNER LOOP STEPS
# ================================================================
# Compare performance with 1, 2, 4, 8 inner loop steps

# # 1 step
# python $BASE_DIR/MAML_trainer.py \
#     --root "$BASE_DIR/Sionna_datasets/ps2_p612/speed5/SISO-UMi/interpolated_noleak" \
#     --device "cuda:0" \
#     --save_init "$INV_DIR/experiments/ablation/1step" \
#     --epoch 2000 \
#     --update_step 1 \
#     --meta_lr 5e-4 \
#     --update_lr 1e-3 \
#     --max_grad_norm 0.5

# # 2 steps
# python $BASE_DIR/MAML_trainer.py \
#     --root "$BASE_DIR/Sionna_datasets/ps2_p612/speed5/SISO-UMi/interpolated_noleak" \
#     --device "cuda:0" \
#     --save_init "$INV_DIR/experiments/ablation/2step" \
#     --epoch 2000 \
#     --update_step 2 \
#     --meta_lr 5e-4 \
#     --update_lr 1e-3 \
#     --max_grad_norm 0.5

# # 4 steps (baseline)
# python $BASE_DIR/MAML_trainer.py \
#     --root "$BASE_DIR/Sionna_datasets/ps2_p612/speed5/SISO-UMi/interpolated_noleak" \
#     --device "cuda:0" \
#     --save_init "$INV_DIR/experiments/ablation/4step" \
#     --epoch 2000 \
#     --update_step 4 \
#     --meta_lr 5e-4 \
#     --update_lr 1e-3 \
#     --max_grad_norm 0.5

# # 8 steps
# python $BASE_DIR/MAML_trainer.py \
#     --root "$BASE_DIR/Sionna_datasets/ps2_p612/speed5/SISO-UMi/interpolated_noleak" \
#     --device "cuda:0" \
#     --save_init "$INV_DIR/experiments/ablation/8step" \
#     --epoch 2000 \
#     --update_step 8 \
#     --meta_lr 5e-4 \
#     --update_lr 1e-3 \
#     --max_grad_norm 0.5

echo "EXPERIMENT 4: Inner loop steps ablation configured"
echo ""

# ================================================================
# EXPERIMENT 5: ABLATION - LEARNING RATE COMBINATIONS
# ================================================================
# Test different meta_lr and task_lr combinations

# # Conservative (1e-4, 1e-3)
# python $BASE_DIR/MAML_trainer.py \
#     --root "$BASE_DIR/Sionna_datasets/ps2_p612/speed5/SISO-UMi/interpolated_noleak" \
#     --device "cuda:0" \
#     --save_init "$INV_DIR/experiments/ablation/lr_conservative" \
#     --epoch 2000 \
#     --meta_lr 1e-4 \
#     --update_lr 1e-3 \
#     --max_grad_norm 0.5

# # Optimal (5e-4, 1e-3) - BASELINE
# python $BASE_DIR/MAML_trainer.py \
#     --root "$BASE_DIR/Sionna_datasets/ps2_p612/speed5/SISO-UMi/interpolated_noleak" \
#     --device "cuda:0" \
#     --save_init "$INV_DIR/experiments/ablation/lr_optimal" \
#     --epoch 2000 \
#     --meta_lr 5e-4 \
#     --update_lr 1e-3 \
#     --max_grad_norm 0.5

# # Aggressive (1e-3, 1e-3)
# python $BASE_DIR/MAML_trainer.py \
#     --root "$BASE_DIR/Sionna_datasets/ps2_p612/speed5/SISO-UMi/interpolated_noleak" \
#     --device "cuda:0" \
#     --save_init "$INV_DIR/experiments/ablation/lr_aggressive" \
#     --epoch 2000 \
#     --meta_lr 1e-3 \
#     --update_lr 1e-3 \
#     --max_grad_norm 0.5

echo "EXPERIMENT 5: Learning rate ablation configured"
echo ""

# ================================================================
# EXPERIMENT 6: SHORT TRAINING RUN (for testing)
# ================================================================
# Quick 100-epoch run to test setup

# python $BASE_DIR/MAML_trainer.py \
#     --root "$BASE_DIR/Sionna_datasets/ps2_p612/speed5/SISO-UMi/interpolated_noleak" \
#     --device "cuda:0" \
#     --save_init "$INV_DIR/experiments/test_run" \
#     --epoch 100 \
#     --n_way 4 \
#     --k_spt 5 \
#     --k_qry 5 \
#     --batchsz 4 \
#     --meta_lr 5e-4 \
#     --update_lr 1e-3 \
#     --update_step 4 \
#     --max_grad_norm 0.5

echo "EXPERIMENT 6: Quick test run configured (100 epochs)"
echo ""

# ================================================================
# POST-PROCESSING: VISUALIZATION AND ANALYSIS
# ================================================================

echo "================================================================"
echo "POST-PROCESSING COMMANDS"
echo "================================================================"
echo ""
echo "After training, use these commands to analyze results:"
echo ""
echo "# Compare training curves:"
echo "python << 'EOF'"
echo "import matplotlib.pyplot as plt"
echo "import numpy as np"
echo ""
echo "experiments = ['baseline', '1step', '2step', '4step', '8step']"
echo "for exp in experiments:"
echo "    losses = np.load(f'experiments/ablation/{exp}/training_losses.npy')"
echo "    plt.plot(losses, label=exp)"
echo ""
echo "plt.xlabel('Epoch')"
echo "plt.ylabel('Loss')"
echo "plt.title('Ablation: Inner Loop Steps')"
echo "plt.legend()"
echo "plt.yscale('log')"
echo "plt.savefig('experiments/ablation/comparison.png')"
echo "EOF"
echo ""

# ================================================================
# NOTES
# ================================================================
echo ""
echo "================================================================"
echo "NOTES"
echo "================================================================"
echo ""
echo "• Uncomment experiments you want to run"
echo "• GPU recommended (training takes ~4 hours for 5000 epochs)"
echo "• Monitor with: watch -n 1 'tail -20 logfile.txt'"
echo "• Results saved in: $INV_DIR/experiments/"
echo "• See EXPERIMENTAL_GUIDE.md for more experiments"
echo ""
echo "Expected Results (5000 epochs, optimal hyperparameters):"
echo "  • Final training loss: ~0.020"
echo "  • 5-shot test MSE: ~0.020"
echo "  • 15-shot test MSE: ~0.012"
echo "  • Training time: ~4 hours (RTX 3090)"
echo ""
echo "Critical Hyperparameters:"
echo "  • max_grad_norm=0.5 (ESSENTIAL - prevents divergence)"
echo "  • meta_lr=5e-4, update_lr=1e-3 (optimal ratio)"
echo "  • update_step=4 (best trade-off)"
echo "  • n_way=4, k_spt=5 (task configuration)"
echo ""
echo "For questions, see:"
echo "  • README.md (complete guide)"
echo "  • INVESTIGATION_SUMMARY.md (quick overview)"
echo "  • CODE_LEVEL_ANALYSIS.md (debugging)"
echo ""
echo "================================================================"

