import os
import numpy as np
import scipy.io as scio
import argparse
import pdb

class Channel_2D:
    
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def open_channel(self, file_path):
        """
        @param file_path: Full path string to the directory that contains channel .mat file matrix.
        """
        with open(file_path, "rb") as f:
            mat_data = scio.loadmat(f)
        return mat_data

    def create_whole_data(self):
        """
        This function processes all .mat files in the input directory,
        storing data in dictionaries with the file name as the key.
        """
        
        data_dict = {}
        label_dict = {}
        rx_signal_dict={}
        tx_signal_dict={}
        
        file_count = 0  

        for root, _, files in os.walk(self.input_dir):
            for file in files:
                # pdb.set_trace()
                if file.endswith('.mat'):
                    file_path = os.path.join(root, file)
                    mat_data = self.open_channel(file_path)
                    
                    
                    data = mat_data['trainData']
                    labels = mat_data['trainLabels']
                    rx_signal= mat_data["rxWaveform"]
                    tx_signal= mat_data["txWaveform"]
                    
                    
                    # Transpose to match the desired shape
                    data = np.transpose(data, (3, 0, 1, 2))
                    labels = np.transpose(labels, (3, 0, 1, 2))
                    
                    # Store in dictionaries with the file name as the key
                    data_dict[file] = data
                    label_dict[file] = labels
                    rx_signal_dict[file] = rx_signal
                    tx_signal_dict[file] = tx_signal
                    
                    
                    
                    file_count += 1  

        print(f"Number of files processed: {file_count}") 
         
        if not data_dict or not label_dict:
            raise ValueError("No data found. Please check the directory path and ensure it contains .mat files.")
        # pdb.set_trace()
        # Save the dictionaries as .npy files
        np.save(os.path.join(self.output_dir, "channel_data_dict.npy"), data_dict)
        np.save(os.path.join(self.output_dir, "channel_label_dict.npy"), label_dict)
        np.save(os.path.join(self.output_dir, "rx_signal_dict.npy"), rx_signal_dict)
        np.save(os.path.join(self.output_dir, "tx_signal_dict.npy"), tx_signal_dict)
        
        
        
        # return data_dict, label_dict

def main(args):
    channel_processor = Channel_2D(args.input_dir, args.output_dir)
    channel_processor.create_whole_data()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process channel .mat files and save as .npy")
    parser.add_argument('--input_dir', type=str, default="/home/CAMPUS/rghasemi/projects/MyPrivaterepo/data", help='Directory containing the .mat files')
    parser.add_argument('--output_dir', type=str, default="/home/CAMPUS/rghasemi/projects/MyPrivaterepo/new_data", help='Directory to save the output .npy files')
    
    args = parser.parse_args()
    main(args)


