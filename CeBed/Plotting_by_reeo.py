# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# def plot_and_save_mse_by_snr(experiment_paths, snr_values, output_dir="plots"):
#     """
#     Reads each experiment CSV, then for each snr in snr_values:
#       • makes a grouped bar chart of mse vs method (columns = experiments)
#       • saves it as plots/mse_snr_<snr>dB.png
#     """
#     # ensure output folder exists
#     os.makedirs(output_dir, exist_ok=True)

#     # load + tag
#     dfs = []
#     for exp_name, csv_path in experiment_paths.items():
#         df = pd.read_csv(csv_path)
#         df['experiment'] = exp_name
#         dfs.append(df)
#     all_df = pd.concat(dfs, ignore_index=True)

#     # loop snrs
#     for snr in snr_values:
#         sub = all_df[all_df['snr'] == snr]
#         pivot = sub.pivot(index='method', columns='experiment', values='mse')

#         # plot
#         fig, ax = plt.subplots()
#         pivot.plot(kind='bar', rot=0, ax=ax)
#         ax.set_title(f'MSE at SNR = {snr} dB')
#         ax.set_xlabel('Method')
#         ax.set_ylabel('MSE')
#         ax.legend(title='Experiment')

#         fig.tight_layout()
#         save_path = os.path.join(output_dir, f"mse_snr_{snr}dB.png")
#         fig.savefig(save_path, dpi=300)
#         print(f"✔ Saved: {save_path}")

#         plt.close(fig)  # free memory

import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_and_save_mse_by_experiment(experiment_paths, snr_values, output_dir="plots_by_exp"):
    """
    For each snr in snr_values:
      • x-axis = experiment (1%, 2%, …, 80% train)
      • one bar per method showing mse
      • saves as plots_by_exp/mse_by_exp_snr_<snr>dB.png
    """
    os.makedirs(output_dir, exist_ok=True)

    # load + tag every experiment
    dfs = []
    for exp_name, csv_path in experiment_paths.items():
        df = pd.read_csv(csv_path)
        df['experiment'] = exp_name
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    # for each snr, pivot so rows=experiments, cols=methods
    for snr in snr_values:
        sub = all_df[all_df['snr'] == snr]
        pivot = sub.pivot(index='experiment', columns='method', values='mse')

        # plot
        fig, ax = plt.subplots()
        pivot.plot(kind='bar', rot=0, ax=ax)
        ax.set_title(f'MSE by Experiment at SNR = {snr} dB')
        ax.set_xlabel('Experiment (% train)')
        ax.set_ylabel('MSE')
        ax.legend(title='Method')

        fig.tight_layout()
        save_path = os.path.join(output_dir, f"mse_by_exp_snr_{snr}dB.png")
        fig.savefig(save_path, dpi=300)
        print(f"✔ Saved: {save_path}")

        plt.close(fig)

if __name__ == "__main__":
    experiment_paths = {
        '1% train':  'train_output_1precent_training/siso_1_umi_block_2_ps2_p612/0/ChannelNet/test_mses.csv',
        '2% train':  'train_output_2precent_training/siso_1_umi_block_2_ps2_p612/0/ChannelNet/test_mses.csv',
        '10% train': 'train_output_10precent_training/siso_1_umi_block_2_ps2_p612/0/ChannelNet/test_mses.csv',
        '60% train': 'train_output_60precent_training/siso_1_umi_block_2_ps2_p612/0/ChannelNet/test_mses.csv',
        '80% train': 'train_output_80precent_training/siso_1_umi_block_2_ps2_p612/0/ChannelNet/test_mses.csv',
    }
    plot_and_save_mse_by_experiment(experiment_paths,
                                    snr_values=[0, 5, 10, 15, 20],
                                    output_dir="plots_by_exp")
