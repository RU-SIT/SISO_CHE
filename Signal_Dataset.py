import glob
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy import io as scio 
import pickle as pkl
import re
import pdb 


class SignalDataset(Dataset):
    
    """
    In this class we should pick signals from different folder with different setup and make images
    """
    def __init__(self, path,  sigID, input_sig_type, output_sig_type, sequence_length, max_tx_antennae=64, max_rx_antenna = 2,tx_wave_dirs=None,rx_wave_dirs=None):
        
        """
            @param tx_wave_dirs (list): This is a list of file paths which governs the files to be loaded in the tx_wave database
            @param rx_wave_dirs (list): This is a list of file paths which governs the files to be loaded in the rx_wave database
        """
        
        self.path = path
        self.sigID = sigID
    
        self.input_sig_type = input_sig_type
        self.output_sig_type = output_sig_type
        
        if tx_wave_dirs is None:
            self.tx_wave_dirs = glob.glob(os.path.join(path, 'TxWave', '*')) 
        # elif len(tx_wave_dirs)!=0:
        #     self.tx_wave_dirs = tx_wave_dirs
        else: 
            self.tx_wave_dirs = self.read_directories(tx_wave_dirs)
         
        if rx_wave_dirs is None:
            assert tx_wave_dirs is None, "both tx_wave_dirs and rx_wave_dirs should be supplied if not neither should be supplied."
            self.rx_wave_dirs = rx_wave_dirs
            
        else:
            self.rx_wave_dirs = [os.path.dirname(os.path.dirname(txfile))+"/RxWave/RxWave_{}".format(os.path.basename(txfile).strip().split("_")[1]) for txfile in self.tx_wave_dirs]
        
        # pdb.set_trace()
        self.TXMAX= max_tx_antennae
        self.RXMAX= max_rx_antenna
        self.sequence_length=sequence_length
        
        self.read_mean_std() #Here we are loading the real and imaginary mean and stadard deviation values from file.
    

    def find_directoris(self, path):
        tx_set = set()
        rx_set = set()
        snr_set = set()
        mod_set = set()
        channel_set = set()
        
        # Template pattern
        pattern = r"TX_(\d+)_RX_(\d+)_SNR_(-?\d+)dB_Mod_(\w+)_([\w-]+)"
        
        for _, dirs, _ in os.walk(path):
            # pdb.set_trace()
            for dir_name in dirs:
                match = re.match(pattern, dir_name)
                if match:
                    tx, rx, snr, mod, channel = match.groups()
                    tx_set.add(int(tx))
                    rx_set.add(int(rx))
                    snr_set.add(int(snr))
                    mod_set.add(mod)
                    channel_set.add(channel)
        
        # Convert sets to lists
        tx_list = list(tx_set)
        rx_list = list(rx_set)
        snr_list = list(snr_set)
        mod_list = list(mod_set)
        channel_list = list(channel_set)
        pdb.set_trace()
        
        return tx_list, rx_list, snr_list, mod_list, channel_list


    def open_signal(self, tx_filepath, rx_filepath) -> dict:
        """
            @param tx_filepath: Full path string to the .mat file containing a TX signal.
            @param rx_filepath: Full path string to the .mat file containing the (CORRESPONDING) RX signal.

       """
        
        with open(tx_filepath,"rb") as f:
            mat1 = scio.loadmat(f)
            
        with open(rx_filepath,"rb") as f:
             mat2 = scio.loadmat(f)
            
        f.close() #Close file descriptor
        org_signal = {'tx': mat1["txWaveform"], 
                      'rx': mat2["rxWaveform"],
                      "tx_channels": mat1["txWaveform"].shape[1], 
                      "rx_channels": mat2["rxWaveform"].shape[1],
                      "seq_length": mat1["txWaveform"].shape[0]}
        
        return org_signal

    def create_subset(self,tensor_padded):
        Xs=list()
        for i in np.arange(0,tensor_padded.size(1),self.sequence_length):
            if i<=(tensor_padded.size(1)-self.sequence_length):
                #size of tens = (#num_receiver_antennae X sequence_length x 2)
                tens=tensor_padded[:,i:i+self.sequence_length]
            # else: (need to figure out how to get the last few time-steps in. currently ignoring)
            #     tens = tensor_padded[:,i:]
            
            Xs.append(tens[None,:])

        Xs=torch.cat(Xs,dim=0)
        Xs = Xs.permute(0,2,1,3)
        
        #size of Xs = (total_time_steps / seq_len , seq_len, max_antennae*2)
        Xs = Xs.reshape((Xs.size(0),Xs.size(1),Xs.size(2)*Xs.size(3))).squeeze()
        
        return Xs


    def read_mean_std(self):
            with open(self.stat_path,"rb") as f:
                stat = pkl.load(f)
    
            # pdb.set_trace()
            self.tx_real_mean = stat['tx']['real_mean']
            self.tx_real_std = stat['tx']['real_std']
            self.tx_real_min = stat['tx']['real_min']
            self.tx_real_max = stat['tx']['real_max']
            self.tx_imag_mean = stat['tx']['imag_mean']
            self.tx_imag_std = stat['tx']['imag_std']
            self.tx_imag_min = stat['tx']['imag_min']
            self.tx_imag_max = stat['tx']['imag_max']
            
            self.rx_real_mean = stat['rx']['real_mean']
            self.rx_real_std = stat['rx']['real_std']
            self.rx_real_min = stat['rx']['real_min']
            self.rx_real_max = stat['rx']['real_max']
            self.rx_imag_mean = stat['rx']['imag_mean']
            self.rx_imag_std = stat['rx']['imag_std']
            self.rx_imag_min = stat['rx']['imag_min']
            self.rx_imag_max = stat['rx']['imag_max']


    def TX_standardize(self, input_real,input_imag):

        #normalize
        input_real_scaled = (input_real - self.tx_real_mean)/(self.tx_real_max-self.tx_real_min)
        input_imag_scaled = (input_imag - self.tx_imag_mean)/(self.tx_imag_max-self.tx_imag_min)
        return input_real_scaled,input_imag_scaled
        
    def _extract_channel_type(self, file_path):
        # Use a regular expression to find the CDL pattern followed by any character
        match = re.search(r'(CDL-[A-E])', file_path)
        if match:
            return match.group(1)  # Returns the matched string, e.g., "CDL-B"
        return None
    
    def _one_hot_encode(self, channel):
        
        channels = ['CDL-A', 'CDL-B', 'CDL-C', 'CDL-D', 'CDL-E']
        index = channels.index(channel)
        onehot = torch.zeros(len(channels))
        onehot[index] = 1
        return onehot
            
    
    def __getitem__(self, index):
        if self.input_sig_type == 'TX' and self.output_sig_type == 'RX':
            
            file_path_tx = self.tx_wave_dirs[index]
            file_path_rx = self.rx_wave_dirs[index] 
        else:
            raise ValueError("Invalid signal type")

        try:            
            signal = self.open_signal(file_path_tx,file_path_rx)
        
        except ValueError:
            print("Value Error Raised for file {}".format(file_path_tx))
            raise ValueError
 
        channel_type = self._extract_channel_type(file_path_tx)
        # Add batch dimension
        onehot = self._one_hot_encode(channel_type).unsqueeze(0)
        # Add sequence dimension
        onehot = onehot.repeat(self.sequence_length, 1)


        transmitter = []
        
        for i in range(signal["tx_channels"]):
            series = signal["tx"][:, i]
            series_real = series.real
            series_imag = series.imag
            series_real_scaled,series_imag_scaled = self.TX_standardize(series_real,series_imag)
            transmitter.append(np.stack([series_real_scaled, series_imag_scaled], axis=-1)) #list of array of each antenna with the lenght of 64 and eacharray has a length of 1530
            

        stacked_tx_signal = np.stack(transmitter, axis=0) #convering a list of arrays to a single array in order to convert to tensor
        tx_tens_signal = torch.from_numpy(stacked_tx_signal).float() #torch.Size([64, 15370, 2])
        
        #pad tensor along the end of the first dimension to ensure that the dimensions equals self.MAX 
        num_tx_antennae = tx_tens_signal.size(0)
        tx_padding = (0, 0, 0, 0, 0,self.TXMAX-num_tx_antennae)
        tx_padded = F.pad(tx_tens_signal, tx_padding, mode='constant', value=0)
          
        reciever = []
        for i in range(signal["rx_channels"]):
            series = signal["rx"][:, i]
            series_real = series.real
            series_imag = series.imag
            # pdb.set_trace()
            
            reciever.append(np.stack([series_real, series_imag], axis=-1))
    
        stacked_rx_signal = np.stack(reciever, axis=0)
        rx_tens_signal = torch.from_numpy(stacked_rx_signal).float()  #create a tensor of size (num_reciever_antennae X num_time_steps X 2)

        num_rx_antennae = rx_tens_signal.size(0)

        rx_padding = (0, 0, 0, 0, 0,self.RXMAX-num_rx_antennae)
        rx_padded = F.pad(rx_tens_signal, rx_padding, mode='constant', value=0) #pad tensor along the end of the first dimension to ensure that the dimensions equals self.MAX 
        
        Tx = self.create_subset(tx_padded)
        Rx = self.create_subset(rx_padded)

        # pdb.set_trace()
        return {'tx':Tx ,
                'rx':Rx ,
                'onehot': onehot,
                "num_tx_antennae":num_tx_antennae,
                "num_rx_antennae":num_rx_antennae,
               "file_path": file_path_tx}

    def __len__(self):

        if self.input_sig_type == 'TX' and self.output_sig_type == 'RX': 
            return len(self.tx_wave_dirs)
        else:
            raise ValueError("Invalid signal type")
