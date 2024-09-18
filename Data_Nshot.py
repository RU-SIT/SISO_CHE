import os
import numpy as np
import torch
import pdb

class ChannelEstimationNShot:

    def __init__(self, root, batchsz, n_way, k_shot, k_query):
        """
        Initializes the dataset for MAML N-shot learning for channel estimation.
        :param root: Path to the root directory containing the channel data.
        :param batchsz: Number of tasks per batch.
        :param n_way: Number of different channel models per task.
        :param k_shot: Number of samples per channel model for training.
        :param k_query: Number of samples per channel model for testing.
        """
        self.batchsz = batchsz
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query

        # Load data and labels from dictionaries
        self.data_dict = np.load(os.path.join(root, 'channel_data_dict.npy'), allow_pickle=True).item()  
        self.labels_dict = np.load(os.path.join(root, 'channel_label_dict.npy'), allow_pickle=True).item()

        # Extract file names to allow random sampling
        self.file_names = list(self.data_dict.keys())

        # Split file names into training and testing sets
        self.train_file_names = self.file_names[:9]
        self.test_file_names = self.file_names[9:]

        self.indexes = {"train": 0, "test": 0}

        # Initialize caches for scaled and unscaled data
        self.scld_datasets_cache = {"train": [], "test": []}
        self.datasets_cache = {"train": [], "test": []}
        self.spt_denom_cache = {"train": [], "test": []}
        self.qry_denom_cache = {"train": [], "test": []}

        # Preload data caches
        self.preload_data()

    def preload_data(self):
        """
        Preloads data caches for both training and testing datasets.
        """
        for mode in ['train', 'test']:
            scld_cache, unscaled_cache, spt_denom, qry_denom = self.load_data_cache(mode)
            self.scld_datasets_cache[mode] = scld_cache
            self.datasets_cache[mode] = unscaled_cache
            self.spt_denom_cache[mode] = spt_denom
            self.qry_denom_cache[mode] = qry_denom

    def unit_scaling(self, data, label):
        """
        Scale each real and imaginary component pixelwise.
        :param data: Data to be scaled.
        :param label: Labels to be scaled.
        :return: Scaled data and labels.
        """
        # Compute the denominator for scaling
        denom = np.sqrt(np.sum(data ** 2, axis=-1, keepdims=True)) + 1e-8  # Avoid division by zero

        # Scale data and labels
        scaled_data = data / denom
        scaled_label = label / denom

        return scaled_data, scaled_label, denom
    
    def load_data_cache(self, mode):
        """
        Preloads a batch of data for N-shot learning.
        :param mode: 'train' or 'test' to specify which dataset to load.
        :return: Two lists containing scaled and unscaled data caches, and two lists containing denominators.
        """
        file_names = self.train_file_names if mode == 'train' else self.test_file_names

        scld_data_cache = []
        data_cache = []
        spt_denom_cache = []
        qry_denom_cache = []

        # Batchify data during training and testing
        for _ in range(self.batchsz if mode == 'train' else 1):  # One task during fine-tuning
            x_qrys, y_qrys, x_spts, y_spts = [], [], [], []
            xs_fixed, ys_fixed = [], []

            for _ in range(1 if mode == 'test' else self.n_way):  # One task in testing mode
                x_qry, y_qry, x_spt, y_spt = [], [], [], []
                x_fixed, y_fixed = [], []

                selected_files = np.random.choice(file_names, 1 if mode == 'test' else self.n_way, replace=False)
                for cur_file in selected_files:
                    data = self.data_dict[cur_file]
                    labels = self.labels_dict[cur_file]

                    # Randomly select samples for support and query sets
                    fixed_samples = data[:10]  # fixed set (first 10)
                    fixed_labels = labels[:10]  # fixed set labels
                    selected_samples = np.random.choice(data.shape[0], 2 * self.k_shot, replace=False)
                    
                    # Query and support sets
                    x_qry.extend(data[selected_samples[:self.k_shot]])
                    y_qry.extend(labels[selected_samples[:self.k_shot]])
                    x_spt.extend(data[selected_samples[self.k_shot:]])
                    y_spt.extend(labels[selected_samples[self.k_shot:]])

                    # Fixed set
                    x_fixed.extend(fixed_samples)
                    y_fixed.extend(fixed_labels)

                # Convert lists to numpy arrays
                x_qry = np.array(x_qry).astype(np.float32)
                y_qry = np.array(y_qry).astype(np.float32)
                x_spt = np.array(x_spt).astype(np.float32)
                y_spt = np.array(y_spt).astype(np.float32)
                x_fixed = np.array(x_fixed).astype(np.float32)
                y_fixed = np.array(y_fixed).astype(np.float32)

                x_qrys.append(x_qry)
                y_qrys.append(y_qry)
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                xs_fixed.append(x_fixed)
                ys_fixed.append(y_fixed)

            # Convert lists to numpy arrays
            x_qrys = np.array(x_qrys)
            y_qrys = np.array(y_qrys)
            x_spts = np.array(x_spts)
            y_spts = np.array(y_spts)
            xs_fixed = np.array(xs_fixed)
            ys_fixed = np.array(ys_fixed)

            # Apply unit scaling
            x_qrys_scld, y_qrys_scld, qry_denom = self.unit_scaling(x_qrys, y_qrys)
            x_spts_scld, y_spts_scld, spts_denom = self.unit_scaling(x_spts, y_spts)
            xs_fixed_scld, ys_fixed_scld, fixed_denom = self.unit_scaling(xs_fixed, ys_fixed)

            # Transpose arrays to match expected dimensions: [batch_size, set_size, channels, height, width]
            x_qrys = x_qrys.transpose(0, 1, 4, 2, 3)
            y_qrys = y_qrys.transpose(0, 1, 4, 2, 3)
            x_spts = x_spts.transpose(0, 1, 4, 2, 3)
            y_spts = y_spts.transpose(0, 1, 4, 2, 3)
            xs_fixed = xs_fixed.transpose(0, 1, 4, 2, 3)
            ys_fixed = ys_fixed.transpose(0, 1, 4, 2, 3)
            x_qrys_scld = x_qrys_scld.transpose(0, 1, 4, 2, 3)
            y_qrys_scld = y_qrys_scld.transpose(0, 1, 4, 2, 3)
            x_spts_scld = x_spts_scld.transpose(0, 1, 4, 2, 3)
            y_spts_scld = y_spts_scld.transpose(0, 1, 4, 2, 3)
            xs_fixed_scld = xs_fixed_scld.transpose(0, 1, 4, 2, 3)
            ys_fixed_scld = ys_fixed_scld.transpose(0, 1, 4, 2, 3)

            # Append to caches
            scld_data_cache.append([x_qrys_scld, y_qrys_scld, x_spts_scld, y_spts_scld, xs_fixed_scld, ys_fixed_scld])
            data_cache.append([x_qrys, y_qrys, x_spts, y_spts, xs_fixed, ys_fixed])
            qry_denom_cache.append(qry_denom)
            spt_denom_cache.append(spts_denom)

        return scld_data_cache, data_cache, qry_denom_cache, spt_denom_cache


    # def load_data_cache(self, mode):
        
    #     """
    #     Preloads a batch of data for N-shot learning.
    #     :param mode: 'train' or 'test' to specify which dataset to load.
    #     :return: Two lists containing scaled and unscaled data caches, and two lists containing denominators.
    #     """
    #     if mode == 'train':
    #         file_names = self.train_file_names

    #         scld_data_cache = []
    #         data_cache = []
    #         spt_denom_cache = []
    #         qry_denom_cache = []

    #         for _ in range(5):  # Five different combinations / episodes
    #             x_qrys, y_qrys, x_spts, y_spts = [], [], [], []
    #             xs_fixed, ys_fixed = [], []
    #             for _ in range(self.batchsz):  # One batch means one set
    #                 x_qry, y_qry, x_spt, y_spt = [], [], [], []
    #                 x_fixed, y_fixed = [], []

    #                 # Randomly select n_way classes (file names)
    #                 selected_files = np.random.choice(file_names, self.n_way, replace=False)
    #                 for cur_file in selected_files:
    #                     data = self.data_dict[cur_file]
    #                     labels = self.labels_dict[cur_file]

    #                     # Randomly select samples for support and query sets
    #                     fixed_samples = data[:10]  # fixed set (first 10)
    #                     fixed_labels = labels[:10]  # fixed set labels
    #                     selected_samples = np.random.choice(data.shape[0], 2*self.k_shot, replace=False)
    #                     x_qry.extend(data[selected_samples[:self.k_shot]])  # Query set
    #                     y_qry.extend(labels[selected_samples[:self.k_shot]])
    #                     x_spt.extend(data[selected_samples[self.k_shot:]])  # Support set
    #                     y_spt.extend(labels[selected_samples[self.k_shot:]])
    #                     x_fixed.extend(fixed_samples)
    #                     y_fixed.extend(fixed_labels)

    #                 # Convert lists to numpy arrays
    #                 x_qry = np.array(x_qry).astype(np.float32)
    #                 y_qry = np.array(y_qry).astype(np.float32)
    #                 x_spt = np.array(x_spt).astype(np.float32)
    #                 y_spt = np.array(y_spt).astype(np.float32)
    #                 x_fixed = np.array(x_fixed).astype(np.float32)
    #                 y_fixed = np.array(y_fixed).astype(np.float32)

    #                 x_qrys.append(x_qry)
    #                 y_qrys.append(y_qry)
    #                 x_spts.append(x_spt)
    #                 y_spts.append(y_spt)
    #                 xs_fixed.append(x_fixed)
    #                 ys_fixed.append(y_fixed)

    #             # Convert lists to numpy arrays
    #             x_qrys = np.array(x_qrys)
    #             y_qrys = np.array(y_qrys)
    #             x_spts = np.array(x_spts)
    #             y_spts = np.array(y_spts)
    #             xs_fixed = np.array(xs_fixed)
    #             ys_fixed = np.array(ys_fixed)

    #             # Apply unit scaling
    #             x_qrys_scld, y_qrys_scld, qry_denom = self.unit_scaling(x_qrys, y_qrys)
    #             x_spts_scld, y_spts_scld, spts_denom = self.unit_scaling(x_spts, y_spts)
    #             xs_fixed_scld, ys_fixed_scld, fixed_denom = self.unit_scaling(xs_fixed, ys_fixed)

    #             # Transpose arrays to match expected dimensions: [batch_size, set_size, channels, height, width]
    #             x_qrys_scld = x_qrys_scld.transpose(0, 1, 4, 2, 3)
    #             y_qrys_scld = y_qrys_scld.transpose(0, 1, 4, 2, 3)
    #             x_spts_scld = x_spts_scld.transpose(0, 1, 4, 2, 3)
    #             y_spts_scld = y_spts_scld.transpose(0, 1, 4, 2, 3)
    #             xs_fixed_scld = xs_fixed_scld.transpose(0, 1, 4, 2, 3)
    #             ys_fixed_scld = ys_fixed_scld.transpose(0, 1, 4, 2, 3)

    #             x_qrys = x_qrys.transpose(0, 1, 4, 2, 3)
    #             y_qrys = y_qrys.transpose(0, 1, 4, 2, 3)
    #             x_spts = x_spts.transpose(0, 1, 4, 2, 3)
    #             y_spts = y_spts.transpose(0, 1, 4, 2, 3)
    #             xs_fixed = xs_fixed.transpose(0, 1, 4, 2, 3)
    #             ys_fixed = ys_fixed.transpose(0, 1, 4, 2, 3)

    #             # Append to caches
    #             scld_data_cache.append([x_qrys_scld, y_qrys_scld, x_spts_scld, y_spts_scld, xs_fixed_scld, ys_fixed_scld])
    #             data_cache.append([x_qrys, y_qrys, x_spts, y_spts, xs_fixed, ys_fixed])
    #             qry_denom_cache.append(qry_denom)
    #             spt_denom_cache.append(spts_denom)

    #     else:
    #         # For testing mode, only use a predefined task
    #         file_names = self.test_file_names
    #         scld_data_cache = []
    #         data_cache = []
    #         spt_denom_cache = []
    #         qry_denom_cache = []

    #         for cur_file in file_names:  # Only one task for fine-tuning
    #             data = self.data_dict[cur_file]
    #             labels = self.labels_dict[cur_file]

    #             fixed_samples = data[:10]
    #             fixed_labels = labels[:10]
    #             selected_samples = np.random.choice(data.shape[0], 2*self.k_shot, replace=False)
    #             x_qry = data[selected_samples[:self.k_shot]]
    #             y_qry = labels[selected_samples[:self.k_shot]]
    #             x_spt = data[selected_samples[self.k_shot:]]
    #             y_spt = labels[selected_samples[self.k_shot:]]

    #             # Convert to numpy arrays
    #             x_qry = np.array(x_qry).astype(np.float32)
    #             y_qry = np.array(y_qry).astype(np.float32)
    #             x_spt = np.array(x_spt).astype(np.float32)
    #             y_spt = np.array(y_spt).astype(np.float32)
    #             x_fixed = np.array(fixed_samples).astype(np.float32)
    #             y_fixed = np.array(fixed_labels).astype(np.float32)

    #             # Apply unit scaling
    #             x_qry_scld, y_qry_scld, qry_denom = self.unit_scaling(x_qry, y_qry)
    #             x_spt_scld, y_spt_scld, spts_denom = self.unit_scaling(x_spt, y_spt)
    #             x_fixed_scld, y_fixed_scld, fixed_denom = self.unit_scaling(x_fixed, y_fixed)
    #             # pdb.set_trace()
                
    #             # Transpose to match expected dimensions
    #             x_qry = x_qry.transpose(0, 3, 1, 2)
    #             y_qry = y_qry.transpose(0, 3, 1, 2)
    #             x_spt = x_spt.transpose(0, 3, 1, 2)
    #             y_spt = y_spt.transpose(0, 3, 1, 2)
    #             x_fixed= x_fixed.transpose(0, 3, 1, 2)
    #             y_fixed= y_fixed.transpose(0, 3, 1, 2)
    #             x_qry_scld = x_qry_scld.transpose(0, 3, 1, 2)
    #             y_qry_scld = y_qry_scld.transpose(0, 3, 1, 2)
    #             x_spt_scld = x_spt_scld.transpose(0, 3, 1, 2)
    #             y_spt_scld = y_spt_scld.transpose(0, 3, 1, 2)
    #             x_fixed_scld = x_fixed_scld.transpose(0, 3, 1, 2)
    #             y_fixed_scld = y_fixed_scld.transpose(0, 3, 1, 2)

    #             scld_data_cache.append([x_qry_scld, y_qry_scld, x_spt_scld, y_spt_scld, x_fixed_scld, y_fixed_scld])
    #             data_cache.append([x_qry, y_qry, x_spt, y_spt, x_fixed, y_fixed])
    #             qry_denom_cache.append(qry_denom)
    #             spt_denom_cache.append(spts_denom)

    #     return scld_data_cache, data_cache, qry_denom_cache, spt_denom_cache

    def next(self, mode='train'):
        """
        Retrieves the next batch from the dataset.
        :param mode: 'train' or 'test'.
        :return: Two tuples containing the next batch of scaled and unscaled data, and two lists of denominators.
        """
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            scld_cache, unscaled_cache, qry_denom_cache, spt_denom_cache = self.load_data_cache(mode)
            self.scld_datasets_cache[mode] = scld_cache
            self.datasets_cache[mode] = unscaled_cache
            self.qry_denom_cache[mode] = qry_denom_cache
            self.spt_denom_cache[mode] = spt_denom_cache

        next_scld_batch = self.scld_datasets_cache[mode][self.indexes[mode]]
        next_unscaled_batch = self.datasets_cache[mode][self.indexes[mode]]
        next_qry_denom = self.qry_denom_cache[mode][self.indexes[mode]]
        next_spt_denom = self.spt_denom_cache[mode][self.indexes[mode]]

        self.indexes[mode] += 1

        return next_scld_batch, next_unscaled_batch, next_qry_denom, next_spt_denom

