import os
import numpy as np
import pdb
import sys
from utils import Utils


class ChannelEstimationNShot:
    def __init__(self, root, batchsz, n_way, k_shot, k_query):
        self.root = root
        self.batchsz = batchsz
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query

        # Load data and labels from dictionaries
        self.data_dict = np.load(os.path.join(root, 'channel_data_dict.npy'), allow_pickle=True).item()  
        self.labels_dict = np.load(os.path.join(root, 'channel_label_dict.npy'), allow_pickle=True).item()
        self.rx_signal = np.load(os.path.join(root, 'rx_signal_dict.npy'), allow_pickle=True).item()  
        self.tx_signal = np.load(os.path.join(root, 'tx_signal_dict.npy'), allow_pickle=True).item()

        self.file_names = list(self.data_dict.keys())
        
        # Determine data type (UMi or TDL) from root path
        if 'UMi' in root or 'UMi' in root.upper():
            self.data_type = 'UMi'
            num_train_channels = 4
        elif 'TDL' in root or 'TDL' in root.upper():
            self.data_type = 'TDL'
            num_train_channels = 10
        else:
            # Default to UMi if not specified
            self.data_type = 'UMi'
            num_train_channels = 4
            print(f"Warning: Could not determine data type from root path '{root}'. Defaulting to UMi with {num_train_channels} channels for training.")
        
        # Split files into train and test based on data type
        self.train_file_names = self.file_names[:num_train_channels]
        self.test_file_names = self.file_names[num_train_channels:]
        
        # Group files by SNR for SNR-based task sampling
        self.train_files_by_snr = self._group_files_by_snr(self.train_file_names)
        self.test_files_by_snr = self._group_files_by_snr(self.test_file_names)
        
        print(f"Data type: {self.data_type}")
        print(f"Training channels: {len(self.train_file_names)} ({', '.join(self.train_file_names)})")
        print(f"Test channels: {len(self.test_file_names)} ({', '.join(self.test_file_names)})")
        
        # Print SNR grouping information
        if self.train_files_by_snr:
            print("\n" + "="*60)
            print("SNR-GROUPED TASKS (Additional Training Tasks)")
            print("="*60)
            print("During training, 3 additional tasks will be created by POOLING")
            print("all channels with the same SNR and sampling from the combined dataset:\n")
            for snr in sorted(self.train_files_by_snr.keys()):
                channel_names = [f.split('_')[-1].replace('.mat', '') for f in self.train_files_by_snr[snr]]
                print(f"  • SNR_{snr}dB_GROUP: Pools {len(self.train_files_by_snr[snr])} channels")
                print(f"    └─ Channels combined: {', '.join(channel_names)}")
            print("="*60 + "\n")

        self.indexes = {"train": 0, "test": 0}
        self.scld_datasets_cache = {"train": [], "test": []}
        self.datasets_cache = {"train": [], "test": []}
        self.qry_name_cache =  {"train": [], "test": []}
        self.spt_name_cache =  {"train": [], "test": []}
        self.fixed_name_cache =  {"train": [], "test": []}
        self.spts_params_cache = {"train": [], "test": []}
        self.qry_params_cache = {"train": [], "test": []}
        self.rx_signals_cache = {"train": [], "test": []}
        self.tx_signals_cache = {"train": [], "test": []}
        # self.perfect_channel_cache = {"train": [], "test": []} 
        self.unique_file_samples = {"train": {}, "test": {}}
        
        self.preload_data()        
    
    def _extract_snr_from_filename(self, filename):
        """
        Extract SNR value from filename.
        Expected format: ...SNR_{value}db... (e.g., SNR_0db, SNR_5db, SNR_10db)
        Returns SNR value as integer, or None if not found.
        """
        import re
        match = re.search(r'SNR[_-]?(\d+)db', filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None
    
    def _group_files_by_snr(self, file_names):
        """
        Group files by their SNR value.
        Returns a dictionary: {snr_value: [list of file names]}
        """
        files_by_snr = {}
        for filename in file_names:
            snr = self._extract_snr_from_filename(filename)
            if snr is not None:
                if snr not in files_by_snr:
                    files_by_snr[snr] = []
                files_by_snr[snr].append(filename)
        return files_by_snr

    def _sample_task_from_file(self, cur_file, mode):
        """
        Create a single task by sampling support/query/fixed/test splits from one channel file.
        Returns tuple containing numpy arrays and metadata for caching.
        """
        data = self.data_dict[cur_file]
        labels = self.labels_dict[cur_file]
        rx_signal = self.rx_signal[cur_file]
        tx_signal = self.tx_signal[cur_file]

        fixed_set_size = 10
        replace = data.shape[0] < (2 * self.k_shot + fixed_set_size)

        fixed_indices = np.arange(min(fixed_set_size, data.shape[0]))
        fixed_samples = data[fixed_indices]
        fixed_labels = labels[fixed_indices]

        available_indices = np.arange(data.shape[0])
        remaining_indices = np.setdiff1d(available_indices, fixed_indices, assume_unique=True)

        if remaining_indices.shape[0] < 2 * self.k_shot:
            selected_indices = np.random.choice(
                available_indices, 2 * self.k_shot, replace=True
            )
        else:
            selected_indices = np.random.choice(
                remaining_indices, 2 * self.k_shot, replace=False
            )

        if cur_file not in self.unique_file_samples[mode]:
            self.unique_file_samples[mode][cur_file] = []
        self.unique_file_samples[mode][cur_file].extend(selected_indices.tolist())
        self.unique_file_samples[mode][cur_file] = list(set(self.unique_file_samples[mode][cur_file]))

        x_qry = data[selected_indices[:self.k_shot]].astype(np.float32)
        y_qry = labels[selected_indices[:self.k_shot]].astype(np.float32)
        x_spt = data[selected_indices[self.k_shot:]].astype(np.float32)
        y_spt = labels[selected_indices[self.k_shot:]].astype(np.float32)

        x_fixed = fixed_samples.astype(np.float32)
        y_fixed = fixed_labels.astype(np.float32)

        selected_set = set(selected_indices.tolist())
        leftover_indices = [idx for idx in available_indices if idx not in selected_set]
        test_data = data[leftover_indices].astype(np.float32)
        test_label = labels[leftover_indices].astype(np.float32)

        x_qry_names = [cur_file] * self.k_shot
        x_spt_names = [cur_file] * self.k_shot
        x_fixed_names = [cur_file] * len(x_fixed)

        return (
            x_qry,
            y_qry,
            x_spt,
            y_spt,
            x_fixed,
            y_fixed,
            test_data,
            test_label,
            x_qry_names,
            x_spt_names,
            x_fixed_names,
            rx_signal,
            tx_signal,
        )
    
    def _sample_snr_grouped_task(self, target_snr, files_by_snr, mode):
        """
        Sample a task from ALL channels with the same SNR value POOLED together.
        This creates grouped tasks where k_query and k_spt are sampled from 
        the combined data of all channels with the target SNR.
        
        For example, for SNR=0dB, this pools:
        TDL-A_0dB + TDL-B_0dB + TDL-C_0dB + TDL-D_0dB + TDL-E_0dB
        and samples from the combined dataset.
        
        Args:
            target_snr: The SNR value to group (0, 5, or 10)
            files_by_snr: Dictionary mapping SNR to file lists
            mode: 'train' or 'test'
            
        Returns: Same format as _sample_task_from_file but with pooled data
        """
        if target_snr not in files_by_snr or len(files_by_snr[target_snr]) == 0:
            raise ValueError(f"No files available for SNR {target_snr}dB")
        
        # Get all files with this SNR
        snr_files = files_by_snr[target_snr]
        
        # Pool all data and labels from these files
        pooled_data = []
        pooled_labels = []
        pooled_rx_signals = []
        pooled_tx_signals = []
        
        for file_name in snr_files:
            pooled_data.append(self.data_dict[file_name])
            pooled_labels.append(self.labels_dict[file_name])
            pooled_rx_signals.append(self.rx_signal[file_name])
            pooled_tx_signals.append(self.tx_signal[file_name])
        
        # Concatenate all data along the sample dimension (axis=0)
        pooled_data = np.concatenate(pooled_data, axis=0)
        pooled_labels = np.concatenate(pooled_labels, axis=0)
        
        # Use first file's rx/tx signals as representative
        rx_signal = pooled_rx_signals[0]
        tx_signal = pooled_tx_signals[0]
        
        # Sample from the pooled data
        fixed_set_size = 10
        fixed_indices = np.arange(min(fixed_set_size, pooled_data.shape[0]))
        fixed_samples = pooled_data[fixed_indices]
        fixed_labels = pooled_labels[fixed_indices]
        
        available_indices = np.arange(pooled_data.shape[0])
        remaining_indices = np.setdiff1d(available_indices, fixed_indices, assume_unique=True)
        
        if remaining_indices.shape[0] < 2 * self.k_shot:
            selected_indices = np.random.choice(
                available_indices, 2 * self.k_shot, replace=True
            )
        else:
            selected_indices = np.random.choice(
                remaining_indices, 2 * self.k_shot, replace=False
            )
        
        # Query and support sets from pooled data
        x_qry = pooled_data[selected_indices[:self.k_shot]].astype(np.float32)
        y_qry = pooled_labels[selected_indices[:self.k_shot]].astype(np.float32)
        x_spt = pooled_data[selected_indices[self.k_shot:]].astype(np.float32)
        y_spt = pooled_labels[selected_indices[self.k_shot:]].astype(np.float32)
        
        # Fixed set
        x_fixed = fixed_samples.astype(np.float32)
        y_fixed = fixed_labels.astype(np.float32)
        
        # Test data - limit to reasonable size to match individual channel tasks
        selected_set = set(selected_indices.tolist())
        leftover_indices = [idx for idx in available_indices if idx not in selected_set]
        
        # Limit test data size to avoid shape mismatches (use same size as typical individual channel)
        # Typical individual channel has ~1000 samples, minus k_shot*2 and fixed_set = ~980 test samples
        max_test_samples = 1000
        if len(leftover_indices) > max_test_samples:
            leftover_indices = np.random.choice(leftover_indices, max_test_samples, replace=False)
        
        test_data = pooled_data[leftover_indices].astype(np.float32)
        test_label = pooled_labels[leftover_indices].astype(np.float32)
        
        # Special name for tracking grouped tasks - easily identifiable
        group_name = f"SNR_{target_snr}dB_GROUP"
        x_qry_names = [group_name] * self.k_shot
        x_spt_names = [group_name] * self.k_shot
        x_fixed_names = [group_name] * len(x_fixed)
        
        return (
            x_qry,
            y_qry,
            x_spt,
            y_spt,
            x_fixed,
            y_fixed,
            test_data,
            test_label,
            x_qry_names,
            x_spt_names,
            x_fixed_names,
            rx_signal,
            tx_signal,
        )

    def preload_data(self):
        # Preload training and testing data caches
        for mode in ['train', 'test']:
            # scld_cache, unscaled_cache, spt_denom, qry_denom, selected_files_cache, rx_signals, tx_signals, perfect_channel_cache, file_names_cache = self.load_data_cache(mode)
            scld_cache, unscaled_cache, qry_name, spt_name, fixed_name, spt_denom, qry_denom, rx_signals, tx_signals  = self.load_data_cache(mode)
            self.scld_datasets_cache[mode] = scld_cache
            self.datasets_cache[mode] = unscaled_cache
            self.qry_name_cache[mode]= qry_name
            self.spt_name_cache[mode] = spt_name
            self.fixed_name_cache[mode]= fixed_name
            self.spts_params_cache[mode] = spt_denom
            self.qry_params_cache[mode] = qry_denom
            self.rx_signals_cache[mode] = rx_signals
            self.tx_signals_cache[mode] = tx_signals
            # self.perfect_channel_cache[mode] = perfect_channel_cache
            
    
    
    def load_data_cache(self, mode):
        """
        Preloads a batch of data for N-shot learning.
        :param mode: 'train' or 'test' to specify which dataset to load.
        :return: Two lists containing scaled and unscaled data caches, and two lists containing denominators.
        Also returns a dictionary of selected files for query, support, and fixed sets.
        """
        file_names = self.train_file_names if mode == 'train' else self.test_file_names
        files_by_snr = self.train_files_by_snr if mode == 'train' else self.test_files_by_snr

        scld_data_cache = []
        data_cache = []
        qry_name_cache = []
        spt_name_cache = []
        fixed_name_cache = []
        qry_params_cache = []
        spts_params_cache = []
        rx_signals_cache = []
        tx_signals_cache = []
        #perfect_channel_cache = []
        unique_samples_cache = {}
        
        # Batchify data during training and testing
        for _ in range(self.batchsz if mode == 'train' else 1):  # One task during fine-tuning
            x_qrys, y_qrys, x_spts, y_spts, ch_qrys, ch_spts= [], [], [], [], [], []
            xs_fixed, ys_fixed = [], []
            x_qry_names, x_spt_names, x_fixed_names = [], [], []
            test_data_all, test_label_all = [], []

            rx_batch = []
            tx_batch = []

            for task_idx in range(1 if mode == 'test' else self.n_way):  # One task in testing mode
                # FIX: Reset lists for EACH task (each task processes ONE file)
                x_qry, y_qry, x_spt, y_spt, ch_qry, ch_spt = [], [], [], [], [], []
                x_fixed, y_fixed = [], []
                test_data = []
                test_label = []
                
                # FIX: Select ONE file per task (not n_way files)
                # For training mode, randomly choose between random sampling and SNR-based grouping
                if mode == 'train' and files_by_snr:
                    # Randomly decide: 50% chance for SNR-based grouping, 50% for random sampling
                    use_snr_grouping = np.random.random() < 0.5
                    
                    if use_snr_grouping:
                        # Select a random SNR value (0, 5, or 10)
                        available_snrs = [snr for snr in [0, 5, 10] if snr in files_by_snr and len(files_by_snr[snr]) > 0]
                        if available_snrs:
                            selected_snr = np.random.choice(available_snrs)
                            snr_file_pool = files_by_snr[selected_snr]
                            # FIX: Select ONE file (not n_way files)
                            cur_file = np.random.choice(snr_file_pool)
                        else:
                            # Fallback to random sampling if no SNR groups available
                            cur_file = np.random.choice(file_names)
                    else:
                        # FIX: Select ONE file (not n_way files)
                        cur_file = np.random.choice(file_names)
                else:
                    # FIX: Select ONE file (not n_way files)
                    cur_file = np.random.choice(file_names)
                
                (
                    x_qry,
                    y_qry,
                    x_spt,
                    y_spt,
                    x_fixed,
                    y_fixed,
                    test_data,
                    test_label,
                    qry_names,
                    spt_names,
                    fixed_names,
                    rx_signal,
                    tx_signal,
                ) = self._sample_task_from_file(cur_file, mode)

                rx_batch.append(rx_signal)
                tx_batch.append(tx_signal)

                x_qrys.append(x_qry)
                y_qrys.append(y_qry)
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                xs_fixed.append(x_fixed)
                ys_fixed.append(y_fixed)
                test_data_all.append(test_data)
                test_label_all.append(test_label)
                x_qry_names.extend(qry_names)
                x_spt_names.extend(spt_names)
                x_fixed_names.extend(fixed_names)

            # Add dedicated SNR-grouped tasks (0, 5, 10 dB) during TDL training
            # These tasks pool ALL channels with the same SNR and sample from the combined dataset
            if mode == 'train' and self.data_type == 'TDL':
                for target_snr in [0, 5, 10]:
                    if target_snr in files_by_snr and len(files_by_snr[target_snr]) > 0:
                        # Sample from pooled data of all channels with this SNR
                        (
                            x_qry,
                            y_qry,
                            x_spt,
                            y_spt,
                            x_fixed,
                            y_fixed,
                            test_data,
                            test_label,
                            qry_names,
                            spt_names,
                            fixed_names,
                            rx_signal,
                            tx_signal,
                        ) = self._sample_snr_grouped_task(target_snr, files_by_snr, mode)

                        rx_batch.append(rx_signal)
                        tx_batch.append(tx_signal)

                        x_qrys.append(x_qry)
                        y_qrys.append(y_qry)
                        x_spts.append(x_spt)
                        y_spts.append(y_spt)
                        xs_fixed.append(x_fixed)
                        ys_fixed.append(y_fixed)
                        test_data_all.append(test_data)
                        test_label_all.append(test_label)
                        x_qry_names.extend(qry_names)
                        x_spt_names.extend(spt_names)
                        x_fixed_names.extend(fixed_names)

            # Normalize test_data sizes to enable proper stacking
            # Find minimum test data size across all tasks
            min_test_size = min(td.shape[0] for td in test_data_all)
            # Truncate all test_data to the minimum size
            test_data_all = [td[:min_test_size] for td in test_data_all]
            test_label_all = [tl[:min_test_size] for tl in test_label_all]
            
            # Convert lists to numpy arrays
            x_qrys = np.array(x_qrys)
            y_qrys = np.array(y_qrys)
            # ch_qrys= np.array(ch_qrys)           
            x_spts = np.array(x_spts)
            y_spts = np.array(y_spts)
            # ch_spts = np.array(ch_spts)
            xs_fixed = np.array(xs_fixed)
            ys_fixed = np.array(ys_fixed)
            test_data_all= np.array(test_data_all)
            test_label_all = np.array(test_label_all)
            # pdb.set_trace()
            
            # Apply unit scaling
            x_qrys_scld, qry_params = Utils.standard_scaling(x_qrys)
            y_qrys_scld, _ = Utils.standard_scaling(y_qrys)
            x_spts_scld, spts_params = Utils.standard_scaling(x_spts)
            y_spts_scld, _ = Utils.standard_scaling(y_spts)
            xs_fixed_scld, xs_fixed_params = Utils.standard_scaling(xs_fixed)
            ys_fixed_scld, _ = Utils.standard_scaling(ys_fixed)
            test_data_scld, test_data_params = Utils.standard_scaling(test_data_all)
            test_label_scld, _ = Utils.standard_scaling(test_label_all)
            
            # pdb.set_trace()
            # Transpose arrays to match expected dimensions: [batch_size, set_size, channels, height, width]
            x_qrys = x_qrys.transpose(0, 1, 4, 2, 3)
            y_qrys = y_qrys.transpose(0, 1, 4, 2, 3)
            # ch_qrys = ch_qrys.transpose(0, 1, 4, 2, 3)
            
            x_spts = x_spts.transpose(0, 1, 4, 2, 3)
            y_spts = y_spts.transpose(0, 1, 4, 2, 3)
            # ch_spts = ch_spts.transpose(0, 1, 4, 2, 3)
            
            xs_fixed = xs_fixed.transpose(0, 1, 4, 2, 3)
            ys_fixed = ys_fixed.transpose(0, 1, 4, 2, 3)
            
            test_data_all = test_data_all.transpose(0, 1, 4, 2, 3)
            test_label_all = test_label_all.transpose(0, 1, 4, 2, 3)
            
            
            x_qrys_scld = x_qrys_scld.transpose(0, 1, 4, 2, 3)
            y_qrys_scld = y_qrys_scld.transpose(0, 1, 4, 2, 3)
            x_spts_scld = x_spts_scld.transpose(0, 1, 4, 2, 3)
            y_spts_scld = y_spts_scld.transpose(0, 1, 4, 2, 3)
            xs_fixed_scld = xs_fixed_scld.transpose(0, 1, 4, 2, 3)
            ys_fixed_scld = ys_fixed_scld.transpose(0, 1, 4, 2, 3)
            test_data_scld=test_data_scld.transpose(0, 1, 4, 2, 3)
            test_label_scld=test_label_scld.transpose(0, 1, 4, 2, 3)
            
            # Append to caches
            scld_data_cache.append([x_qrys_scld, y_qrys_scld, x_spts_scld, y_spts_scld, xs_fixed_scld, ys_fixed_scld, test_data_scld, test_label_scld])
            data_cache.append([x_qrys, y_qrys, x_spts, y_spts, xs_fixed, ys_fixed, test_data_all,test_label_all ])
            qry_name_cache.append(x_qry_names)
            spt_name_cache.append(x_spt_names)
            fixed_name_cache.append(x_fixed_names)
            qry_params_cache.append(qry_params)
            spts_params_cache.append(spts_params)
            # pdb.set_trace()
            rx_signals_cache.append(rx_batch)
            tx_signals_cache.append(tx_batch)
            # perfect_channel_cache.append([ch_qrys, ch_spts])
        # pdb.set_trace()
        
        # return scld_data_cache, data_cache, qry_denom_cache, spt_denom_cache, selected_files_cache, rx_signals_cache, tx_signals_cache, perfect_channel_cache, file_names_cache
        
        return scld_data_cache, data_cache, qry_name_cache, spt_name_cache, fixed_name_cache, spts_params_cache, qry_params_cache, rx_signals_cache, tx_signals_cache
    
    def next(self, mode='train'):
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            (scld_cache, unscaled_cache, qry_name_cache, spt_name_cache, fixed_name_cache,
            spts_params_cache, qry_params_cache, rx_signals_cache, tx_signals_cache) = self.load_data_cache(mode)
            self.scld_datasets_cache[mode] = scld_cache
            self.datasets_cache[mode]      = unscaled_cache
            self.qry_name_cache[mode]      = qry_name_cache
            self.spt_name_cache[mode]      = spt_name_cache
            self.fixed_name_cache[mode]    = fixed_name_cache
            self.qry_params_cache[mode]    = qry_params_cache
            self.spts_params_cache[mode]   = spts_params_cache
            self.rx_signals_cache[mode]    = rx_signals_cache
            self.tx_signals_cache[mode]    = tx_signals_cache

        idx = self.indexes[mode]

        next_scld_batch   = self.scld_datasets_cache[mode][idx]
        next_unscaled_batch = self.datasets_cache[mode][idx]
        next_qry_name     = self.qry_name_cache[mode][idx]
        next_spt_name     = self.spt_name_cache[mode][idx]
        next_fixed_name   = self.fixed_name_cache[mode][idx]
        next_qry_params   = self.qry_params_cache[mode][idx]
        next_spts_params  = self.spts_params_cache[mode][idx]
        next_rx_signal    = self.rx_signals_cache[mode][idx]
        next_tx_signal    = self.tx_signals_cache[mode][idx]

        self.indexes[mode] += 1

        return (next_scld_batch, next_unscaled_batch,
                next_qry_name, next_spt_name, next_fixed_name,
                next_qry_params, next_spts_params,
                next_rx_signal, next_tx_signal)

