import torch
import glob
import os
from torch.utils.data import Dataset, DataLoader
from pytorch_nnue.utils import get_nstm_indices, sanitize_halfkp_indices

class HalfKPDataset(Dataset):
    def __init__(self, data_dir='data/halfkp_data'):
        """Initialization"""
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        self.total_samples = 0
        self.samples_per_file = []
        self.current_data = None
        
        for path in self.file_paths:
            sz = path.split("_")[-1].split(".")[0][2:]  # Extract size from filename
            self.total_samples += int(sz)
            self.samples_per_file.append(int(sz))
        
        self.cumulative_sizes = torch.tensor(self.samples_per_file).cumsum(dim=0)
        self.current_file_idx = -1

    def __len__(self):
        """Denotes the total number of samples"""
        return self.total_samples   

    def _load_file(self, file_idx):
        if file_idx != self.current_file_idx:
            self.current_data = torch.load(self.file_paths[file_idx])
            self.current_file_idx = file_idx

    def __getitem__(self, idx):
        """Generates one sample of data"""
        file_idx = torch.searchsorted(self.cumulative_sizes, idx, right=True)
        if file_idx > 0:
            local_idx = idx - self.cumulative_sizes[file_idx - 1].item()
        else:
            local_idx = idx
            
        self._load_file(file_idx)
        
        # 2. Extraire les données STM (Joueur)
        stm_indices = self.current_data['indices'][local_idx]#sanitize_halfkp_indices(self.current_data['indices'][local_idx])
        nstm_king_sq = self.current_data['nstm_kings'][local_idx]
        # stm_king_sq = self.current_data['stm_kings'][local_idx]
        # nstm_indices =get_nstm_indices(stm_indices, nstm_king_sq) # sanitize_halfkp_indices(get_nstm_indices(stm_indices, nstm_king_sq))
        
        return {
            'stm_indices': stm_indices,
            'nstm_kings': nstm_king_sq,
            'wdl': torch.as_tensor(self.current_data['wdl'][local_idx], dtype=torch.float32),
            'score': torch.as_tensor(self.current_data['score'][local_idx], dtype=torch.float32),
        }
        