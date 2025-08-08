import os
import numpy as np
import pdb
from utils import Utils
from scipy.io import loadmat


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
        # pdb.set_trace()
        self.train_file_names = self.file_names[:4]
        self.test_file_names = self.file_names[4:]

        self.indexes = {"train": 0, "test": 0}
        self.scld_datasets_cache = {"train": [], "test": []}
        self.datasets_cache = {"train": [], "test": []}
        self.qry_name_cache =  {"train": [], "test": []}
        self.spt_name_cache =  {"train": [], "test": []}
        self.fixed_name_cache =  {"train": [], "test": []}
        self.spt_denom_cache = {"train": [], "test": []}
        self.qry_denom_cache = {"train": [], "test": []}
        self.rx_signals_cache = {"train": [], "test": []}
        self.tx_signals_cache = {"train": [], "test": []}
        # self.perfect_channel_cache = {"train": [], "test": []} 
        self.unique_file_samples = {"train": {}, "test": {}}
        
        self.preload_data()        
   

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
            self.spt_denom_cache[mode] = spt_denom
            self.qry_denom_cache[mode] = qry_denom
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

        scld_data_cache = []
        data_cache = []
        qry_name_cache = []
        spt_name_cache = []
        fixed_name_cache = []
        spt_denom_cache = []
        qry_denom_cache = []
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

            for _ in range(1 if mode == 'test' else self.n_way):  # One task in testing mode
                x_qry, y_qry, x_spt, y_spt, ch_qry, ch_spt = [], [], [], [], [], []
                x_fixed, y_fixed = [], []
                test_data = []
                test_label = []
                selected_files = np.random.choice(file_names, 1 if mode == 'test' else self.n_way, replace=True)
                

                for cur_file in selected_files:
                    data = self.data_dict[cur_file]
                    labels = self.labels_dict[cur_file]
                    rx_signal = self.rx_signal[cur_file]
                    tx_signal = self.tx_signal[cur_file]
                    # p_channel = self.noiseless_channel( self.root, cur_file, label_prefix="perfect_channel_SNR200_")
                    # pdb.set_trace()
                    # self.root
                    
                    rx_batch.append(rx_signal)
                    tx_batch.append(tx_signal)

                    # Randomly select samples for support and query sets
                    fixed_samples = data[:10]  # fixed set (first 10)
                    fixed_labels = labels[:10]  # fixed set labels
                    selected_samples = np.random.choice(data.shape[0], 2 * self.k_shot, replace=False)
                    
                    if cur_file not in self.unique_file_samples[mode]:
                        self.unique_file_samples[mode][cur_file] = []
                    self.unique_file_samples[mode][cur_file].extend(selected_samples.tolist())
                    self.unique_file_samples[mode][cur_file] = list(set(self.unique_file_samples[mode][cur_file]))
                    # Query and support sets
                    x_qry.extend(data[selected_samples[:self.k_shot]])
                    y_qry.extend(labels[selected_samples[:self.k_shot]])
                    # ch_qry.extend(p_channel[selected_samples[:self.k_shot]])
                    x_spt.extend(data[selected_samples[self.k_shot:]])
                    y_spt.extend(labels[selected_samples[self.k_shot:]])
                    # ch_spt.extend(p_channel[selected_samples[self.k_shot:]])
                    
                    # Fixed set
                    x_fixed.extend(fixed_samples)
                    y_fixed.extend(fixed_labels)
                    
                    x_qry_names.extend([cur_file] * self.k_shot)
                    # y_qry_names.extend([cur_file] * self.k_shot)
                    x_spt_names.extend([cur_file] * self.k_shot)
                    # y_spt_names.extend([cur_file] * self.k_shot)
                    
                    x_fixed_names.extend([cur_file] * len(fixed_samples))
                    # y_fixed_names.extend([cur_file] * len(fixed_samples))
                    

                    all_indecies= set(range(data.shape[0]))
                    unselected_samples= list(all_indecies- set(selected_samples))
                    test_data.extend(data[unselected_samples])
                    test_label.extend(data[unselected_samples])
                         

                    
                # pdb.set_trace()
                # Convert lists to numpy arrays
                x_qry = np.array(x_qry).astype(np.float32)
                y_qry = np.array(y_qry).astype(np.float32)
                x_spt = np.array(x_spt).astype(np.float32)
                y_spt = np.array(y_spt).astype(np.float32)
                # ch_qry = np.array(ch_qry).astype(np.float32)
                # ch_spt = np.array(ch_spt).astype(np.float32)
                x_fixed = np.array(x_fixed).astype(np.float32)
                y_fixed = np.array(y_fixed).astype(np.float32)
                test_data = np.array(test_data).astype(np.float32)
                test_label = np.array(test_label).astype(np.float32)
                

                x_qrys.append(x_qry)
                y_qrys.append(y_qry)
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                # ch_qrys.append(ch_qry)
                # ch_spts.append(ch_spt)
                xs_fixed.append(x_fixed)
                ys_fixed.append(y_fixed)
                test_data_all.append(test_data)
                # pdb.set_trace()
                
                test_label_all.append(test_label)
                # pdb.set_trace()
                
                

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
            x_qrys_scld, y_qrys_scld, qry_denom = Utils.unit_scaling(x_qrys, y_qrys)
            x_spts_scld, y_spts_scld, spts_denom = Utils.unit_scaling(x_spts, y_spts)
            xs_fixed_scld, ys_fixed_scld, fixed_denom = Utils.unit_scaling(xs_fixed, ys_fixed)
            test_data_scld, test_label_scld, test_denom = Utils.unit_scaling(test_data_all, test_label_all)
            
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
            qry_denom_cache.append(qry_denom)
            spt_denom_cache.append(spts_denom)
            # pdb.set_trace()
            rx_signals_cache.append(rx_batch)
            tx_signals_cache.append(tx_batch)
            # perfect_channel_cache.append([ch_qrys, ch_spts])
        # pdb.set_trace()
        
        # return scld_data_cache, data_cache, qry_denom_cache, spt_denom_cache, selected_files_cache, rx_signals_cache, tx_signals_cache, perfect_channel_cache, file_names_cache
        
        return scld_data_cache, data_cache, qry_name_cache, spt_name_cache, fixed_name_cache, qry_denom_cache, spt_denom_cache, rx_signals_cache, tx_signals_cache

    def next(self, mode='train'):
        """
        Retrieves the next batch from the dataset.
        """
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            scld_cache, unscaled_cache, qry_name_cache, spt_name_cache, fixed_name_cache, qry_denom_cache, spt_denom_cache, rx_signals_cache, tx_signals_cache = self.load_data_cache(mode)
            self.scld_datasets_cache[mode] = scld_cache
            self.datasets_cache[mode] = unscaled_cache
            self.qry_name_cache[mode]= qry_name_cache
            self.spt_name_cache[mode] = spt_name_cache
            self.fixed_name_cache[mode]= fixed_name_cache
            self.qry_denom_cache[mode] = qry_denom_cache
            self.spt_denom_cache[mode] = spt_denom_cache
            self.rx_signals_cache[mode] = rx_signals_cache
            self.tx_signals_cache[mode] = tx_signals_cache

            # self.perfect_channel_cache[mode] = perfect_channel_cache

        next_scld_batch = self.scld_datasets_cache[mode][self.indexes[mode]]
        next_unscaled_batch = self.datasets_cache[mode][self.indexes[mode]]
        next_qry_name = self.qry_name_cache[mode][self.indexes[mode]]
        next_spt_name = self.spt_name_cache[mode][self.indexes[mode]]
        next_fixed_name = self.fixed_name_cache[mode][self.indexes[mode]]
        next_qry_denom = self.qry_denom_cache[mode][self.indexes[mode]]
        next_spt_denom = self.spt_denom_cache[mode][self.indexes[mode]]
        next_rx_signal = self.rx_signals_cache[mode]
        next_tx_signal = self.tx_signals_cache[mode]
        # next_perfect_channel_cache = self.perfect_channel_cache[mode][self.indexes[mode]]


        self.indexes[mode] += 1

        # return next_scld_batch, next_unscaled_batch, next_qry_denom, next_spt_denom, next_rx_signal, next_tx_signal, next_perfect_channel_cache, next_file_names
        return next_scld_batch, next_unscaled_batch, next_qry_name, next_spt_name, next_fixed_name, next_qry_denom, next_spt_denom, next_rx_signal, next_tx_signal

    # def save_unique_samples(self, save_path):
    #         """
    #         Save the unique file-to-sample mappings to a file.
    #         """
    #         np.save(os.path.join(save_path, "unique_file_samples.npy"), self.unique_file_samples)