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
        file_count = 0  

        for root, _, files in os.walk(self.input_dir):
            for file in files:
                pdb.set_trace()
                if file.endswith('.mat'):
                    file_path = os.path.join(root, file)
                    mat_data = self.open_channel(file_path)
                    
                    
                    data = mat_data['trainData']
                    labels = mat_data['trainLabels']
                    
                    # Transpose to match the desired shape
                    data = np.transpose(data, (3, 0, 1, 2))
                    labels = np.transpose(labels, (3, 0, 1, 2))
                    
                    # Store in dictionaries with the file name as the key
                    data_dict[file] = data
                    label_dict[file] = labels
                    
                    file_count += 1  

        print(f"Number of files processed: {file_count}") 
         
        if not data_dict or not label_dict:
            raise ValueError("No data found. Please check the directory path and ensure it contains .mat files.")
        pdb.set_trace()
        # Save the dictionaries as .npy files
        np.save(os.path.join(self.output_dir, "channel_data_dict.npy"), data_dict)
        np.save(os.path.join(self.output_dir, "channel_label_dict.npy"), label_dict)
        
        return data_dict, label_dict

def main(args):
    channel_processor = Channel_2D(args.input_dir, args.output_dir)
    channel_processor.create_whole_data()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process channel .mat files and save as .npy")
    parser.add_argument('--input_dir', type=str, help='Directory containing the .mat files')
    parser.add_argument('--output_dir', type=str, help='Directory to save the output .npy files')
    
    args = parser.parse_args()
    main(args)


# import os
# import numpy as np
# import scipy.io as scio
# import argparse


# class Channel_2D:
    
#     def __init__(self, input_dir, output_dir):
#         self.input_dir = input_dir
#         self.output_dir = output_dir

#     def open_channel(self, file_path):
#         """
#         @param file_path: Full path string to the directory that contains channel .mat file matrix.
#         dict_keys(['__header__', '__version__', '__globals__', 'K', 'L', 'N0', 'R', 'SNR', 'SNRdB', 'None', 'chInfo', 'dmrsIndices',
#         'dmrsSymbols', 'estChannelGrid', 'estChannelGridPerfect', 'interpChannelGrid', 'maxChDelay', 'nnInput', 'noise', 'offset', 'pathFilters',
#         'pathGains', 'pdschGrid', 'rxGrid', 'rxWaveform', 'sampleTimes','simParameters', 'trainData', 'trainLabels', 'trainModel',
#         'txWaveform', 'waveformInfo', '__function_workspace__'])
#         """
#         with open(file_path, "rb") as f:
#             mat1 = scio.loadmat(f)
#         return mat1

#     def create_whole_data(self):
#         """_summary_
#         This function is used for creating an npy file of all data.
#         Output is 2 npy files each with size of [15, 256, 1, 612, 14, 2]
#         """
        
#         data_channels = []
#         label_channels = []
#         channel_name=[]
#         file_count = 0  

#         for root, _, files in os.walk(self.input_dir):
#             for file in files:
#                 if file.endswith('.mat'):
#                     file_path = os.path.join(root, file)
#                     mat_data = self.open_channel(file_path)
#                     data = mat_data['trainData']
#                     labels = mat_data['trainLabels']
#                     data = np.transpose(data, (3, 0, 1, 2))
#                     labels = np.transpose(labels, (3, 0, 1, 2))
#                     data_channels.append(data)
#                     label_channels.append(labels)
#                     file_count += 1  

#         print(f"Number of files processed: {file_count}") 
         
#         if not data_channels or not label_channels:
#             raise ValueError("No data found. Please check the directory path and ensure it contains .mat files.")

#         # Stack the data along a new dimension
#         data_channels = np.stack(data_channels, axis=0)
#         label_channels = np.stack(label_channels, axis=0)
        
#         np.save(os.path.join(self.output_dir, "channel_data.npy"), data_channels)
#         np.save(os.path.join(self.output_dir, "channel_label.npy"), label_channels)
        
#         return data_channels, label_channels

# def main(args):
#     channel_processor = Channel_2D(args.input_dir, args.output_dir)
#     channel_processor.create_whole_data()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Process channel .mat files and save as .npy")
#     parser.add_argument('input_dir', type=str, help='Directory containing the .mat files')
#     parser.add_argument('output_dir', type=str, help='Directory to save the output .npy files')
    
#     args = parser.parse_args()
#     main(args)
