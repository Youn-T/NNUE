import torch
import glob
import os
import random
from torch.utils.data import IterableDataset, get_worker_info
from sf13_nnue.utils import get_nstm_indices, centipawn_to_prob

class HalfKPDataset(IterableDataset):
    def __init__(self, data_dir='data/halfkp_data', batch_size=8192, shuffle=True):
        """Initialization"""
        super().__init__()
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._length = None

    def __len__(self):
        if self._length is None:
            total_samples = 0
            for path in self.file_paths:
                # Extraire la taille à partir du nom du fichier wX_Y_szXXXX.pt
                filename = os.path.basename(path)
                try:
                    # Trouve le bouton 'sz', prend ce qui suit et retire ce qui n'est pas un chiffre
                    import re
                    match = re.search(r'sz(\d+)', filename)
                    if match:
                        total_samples += int(match.group(1))
                except (ValueError, IndexError):
                    pass
            # Le nombre de batches total est la somme des échantillons divisée par batch_size
            self._length = (total_samples + self.batch_size - 1) // self.batch_size
        return self._length
        
    def __iter__(self):
        worker_info = get_worker_info()
        
        # 1. Sharding files per worker to avoid duplicate data and contention
        if worker_info is None:
            files_to_process = self.file_paths
        else:
            files_to_process = [
                self.file_paths[i] for i in range(len(self.file_paths))
                if i % worker_info.num_workers == worker_info.id
            ]

        if self.shuffle:
            files_to_process = list(files_to_process)
            random.shuffle(files_to_process)
            
        for path in files_to_process:
            # 2. Load the entire file tensor at once into memory (orders of magnitude faster)
            data = torch.load(path, weights_only=True)
            indices = data['indices'].to(torch.long) 
            nstm_kings = data['nstm_kings'].to(torch.long)
            wdl = data['wdl'].to(torch.float32)
            score = data['score'].to(torch.float32)
            wdl = torch.ones_like(wdl) - wdl # /!\ Provisoire ! A inverser si l'on regénère le dataset, corrige le bug  d'inversion du label WDL
            score = data['score'].to(torch.float32)
            
            # Compute nstm_indices fully vectorized over the whole chunk!
            stm_indices = indices
            # Uncomment if you need nstm_indices for the model (usually required for evaluation):
            # nstm_indices = get_nstm_indices(stm_indices, nstm_kings)
            
            n_samples = stm_indices.size(0)
            
            # 3. Shuffle chunks internally
            if self.shuffle:
                perm = torch.randperm(n_samples)
                stm_indices = stm_indices[perm]
                nstm_kings = nstm_kings[perm]
                wdl = wdl[perm]
                score = score[perm]
                # if nstm_indices is generated, shuffle it too
                # nstm_indices = nstm_indices[perm]
                
            # 4. Yield pre-batched sliced tensors directly to bypass any collate overhead
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                
                batch_stm = stm_indices[start_idx:end_idx]
                batch_nstm_kings = nstm_kings[start_idx:end_idx]
                batch_wdl = wdl[start_idx:end_idx]
                batch_score = score[start_idx:end_idx]
                # batch_score = batch_score#centipawn_to_prob(batch_score)  # Convert centipawns to probabilities
                batch_nstm = get_nstm_indices(batch_stm, batch_nstm_kings)
                
                batch_stm = torch.where(batch_stm == -1, 40960, batch_stm)
                batch_nstm = torch.where(batch_nstm == -1, 40960, batch_nstm)

                yield (batch_stm, batch_nstm), (batch_score, batch_wdl)

        