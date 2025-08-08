import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import argparse
from utils import Utils

class FewShotDataset(Dataset):
    def __init__(self, data, labels, n_way, k_shot, k_query, snr_values):
        self.data = data
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.classes = np.unique(labels)
        self.snr_values = snr_values

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, idx):
        # Select n_way classes
        selected_classes = np.random.choice(self.classes, self.n_way, replace=False)
        support_set = []
        query_set = []
        snr_set = []

        for cls in selected_classes:
            # Get all samples of the class
            cls_indices = np.where(self.labels == cls)[0]
            cls_samples = self.data[cls_indices]

            # Randomly select k_shot + k_query samples
            selected_indices = np.random.choice(cls_samples.shape[0], self.k_shot + self.k_query, replace=False)
            support_set.append(cls_samples[selected_indices[:self.k_shot]])
            query_set.append(cls_samples[selected_indices[self.k_shot:]])
            snr_set.append(self.snr_values[cls])  # Assign SNR value

        support_set = np.concatenate(support_set, axis=0)
        query_set = np.concatenate(query_set, axis=0)
        snr_set = np.array(snr_set)

        return torch.tensor(support_set), torch.tensor(query_set), torch.tensor(snr_set)

class HDF5DataLoader:
    def __init__(self, file_path, batchsz, n_way, k_shot, k_query):
        self.file_path = file_path
        self.batchsz = batchsz
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query

        # Load data from HDF5 file
        with h5py.File(self.file_path, 'r') as f:
            self.data = f['data'][:]  # Assuming 'data' is the dataset name
            self.labels = f['labels'][:]  # Assuming 'labels' is the dataset name

        # Split domains
        self.num_domains = self.data.shape[0]
        self.train_domains = list(range(self.num_domains - 1))
        self.finetune_domain = self.num_domains - 1

        # Assign SNR values to each domain
        self.snr_range = np.arange(0, 25, 5)
        self.snr_values = {domain: snr for domain, snr in zip(self.train_domains, self.snr_range)}

    def get_batch(self, mode='train'):
        if mode == 'train':
            selected_domains = np.random.choice(self.train_domains, self.n_way, replace=False)
        else:
            selected_domains = [self.finetune_domain]

        x_qrys, y_qrys, x_spts, y_spts, snr_values = [], [], [], [], []

        for domain in selected_domains:
            data = self.data[domain]
            labels = self.labels[domain]

            # Randomly select samples for support and query sets
            selected_samples = np.random.choice(data.shape[0], self.k_shot + self.k_query, replace=False)
            x_spt = data[selected_samples[:self.k_shot]]
            y_spt = labels[selected_samples[:self.k_shot]]
            x_qry = data[selected_samples[self.k_shot:]]
            y_qry = labels[selected_samples[self.k_shot:]]

            x_qrys.append(x_qry)
            y_qrys.append(y_qry)
            x_spts.append(x_spt)
            y_spts.append(y_spt)
            snr_values.append(self.snr_values[domain])  # Append SNR value

        # Convert lists to numpy arrays
        x_qrys = np.array(x_qrys).astype(np.float32)
        y_qrys = np.array(y_qrys).astype(np.float32)
        x_spts = np.array(x_spts).astype(np.float32)
        y_spts = np.array(y_spts).astype(np.float32)
        snr_values = np.array(snr_values).astype(np.float32)

        # Apply unit scaling
        x_qrys_scld, y_qrys_scld, _ = Utils.unit_scaling(x_qrys, y_qrys)
        x_spts_scld, y_spts_scld, _ = Utils.unit_scaling(x_spts, y_spts)

        return x_qrys_scld, y_qrys_scld, x_spts_scld, y_spts_scld, snr_values

def create_dataloader(hdf5_file, n_way, k_shot, k_query, batch_size):
    with h5py.File(hdf5_file, 'r') as f:
        # Assuming the HDF5 file has datasets named 'data' and 'labels'
        data = f['data'][:]
        labels = f['labels'][:]

        # Split domains
        unique_domains = np.unique(labels)
        fine_tune_domain = np.random.choice(unique_domains)
        train_domains = unique_domains[unique_domains != fine_tune_domain]

        # Create datasets for training
        train_data = data[np.isin(labels, train_domains)]
        train_labels = labels[np.isin(labels, train_domains)]

        # Create the few-shot dataset
        train_dataset = FewShotDataset(train_data, train_labels, n_way, k_shot, k_query, {domain: snr for domain in train_domains})

        # Create DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, fine_tune_domain

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a DataLoader for MAML training.")
    parser.add_argument('--hdf5_file', type=str, required=True, help='Path to the HDF5 file.')
    parser.add_argument('--n_way', type=int, default=5, help='Number of classes per task.')
    parser.add_argument('--k_shot', type=int, default=5, help='Number of support samples per class.')
    parser.add_argument('--k_query', type=int, default=15, help='Number of query samples per class.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for DataLoader.')

    args = parser.parse_args()

    train_loader, fine_tune_domain = create_dataloader(
        args.hdf5_file, args.n_way, args.k_shot, args.k_query, args.batch_size
    )

    print(f"Fine-tune domain: {fine_tune_domain}")
    print(f"Number of tasks in DataLoader: {len(train_loader)}")



#  python create_dataloader.py --hdf5_file data.hdf5 --n_way 5 --k_shot 5 --k_query 15 --batch_size 32