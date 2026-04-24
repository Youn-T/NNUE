import torch
import torch.nn.functional as F

HALFKP_NUM_EMBEDDINGS = 40960

class CReLU(torch.nn.Module):
    def __init__(self, clip_value=255.0):
        super().__init__()
        self.clip_value = clip_value

    # @staticmethod
    def forward(self, x):
        return torch.clamp(x, min=0.0, max=self.clip_value)

    
def hybrid_loss(pred, score, WDL, alpha=0.005):
    mse_loss = F.mse_loss(centipawn_to_prob(pred, scale=400.0), centipawn_to_prob(score, scale=400.0))
    bce_loss = F.binary_cross_entropy_with_logits(pred / 400.0, WDL.float())
    return alpha * mse_loss + (1 - alpha) * bce_loss

def mse_loss(pred, target):
    return F.mse_loss(centipawn_to_prob(pred, scale=400.0), centipawn_to_prob(target, scale=400.0))  
    
def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.EmbeddingBag):
        # Initialisation plus douce pour éviter l'explosion des gradients
        torch.nn.init.uniform_(m.weight, a=-0.01, b=0.01)
        # Assure-toi que le padding index reste bien à zéro !
        with torch.no_grad():
            m.weight[40960].fill_(0.0)
            
            
def get_nstm_indices(indices_white, black_king_sq):
    """
    Version vectorisée pour PyTorch.
    indices_white: Tensor de forme (N, 30) ou (30,)
    black_king_sq: Tensor de forme (N, 1) ou (N,) ou un int
    """
    # 1. On isole la partie "Pièce + Case" (0-639)
    remainder = indices_white % 640
    
    # 2. Extraire PieceOffset (0-9) et PieceSq (0-63)
    p_idx_w = torch.div(remainder, 64, rounding_mode='floor')
    p_sq_w = remainder % 64
    
    # 3. Transformer pour la perspective Noire
    # Swap des offsets (0-4 <-> 5-9)
    p_idx_b = (p_idx_w + 5) % 10
    # Flip vertical de la case de la pièce (XOR 56)
    p_sq_b = p_sq_w ^ 56
    
    # 4. Préparer la case du Roi Noir (vue par le noir : flip vertical)
    # On s'assure que black_king_sq est un tenseur pour le broadcasting
    if not isinstance(black_king_sq, torch.Tensor):
        black_king_sq = torch.tensor(black_king_sq, device=indices_white.device)
    
    k_sq_b_view = black_king_sq ^ 56

    # Si black_king_sq est (N,), on le transforme en (N, 1) pour le multiplier
    # correctement avec indices_white qui est (N, 30)
    if k_sq_b_view.dim() == 1 and indices_white.dim() == 2:
        k_sq_b_view = k_sq_b_view.unsqueeze(-1)
    
    # 5. Reconstruire l'indice final
    indices_black = (k_sq_b_view * 640) + (p_idx_b * 64) + p_sq_b
    
    # Restaurer le padding à -1
    indices_black = torch.where(indices_white == -1, -1, indices_black)
    
    # S'assurer que les indices convertis restent valides entre 0 et 40959 
    # Mettre à 0 ou -1 pour éviter l'erreur cuda "index out of range"
    indices_black = torch.where((indices_black >= 40960) | (indices_black < -1), -1, indices_black)
    
    return indices_black


def sanitize_halfkp_indices(indices, num_embeddings=HALFKP_NUM_EMBEDDINGS):
    """
    Sanitize HalfKP indices for EmbeddingBag.

    Some legacy chunks may have been persisted as int16, which wraps values
    above 32767 to negative numbers. This restores wrapped values and ensures
    all indices are inside [0, num_embeddings).
    """
    # indices = torch.as_tensor(indices, dtype=torch.long)

    # # Recover values wrapped by signed int16 storage (two's complement).
    # indices = torch.where(indices < 0, indices + 65536, indices)

    # # Keep runtime safe even if a chunk still contains unexpected outliers.
    # return indices.clamp_(0, num_embeddings - 1)
    mask = (indices != -1).all(dim=1)  # True if row has no -1
    return indices[mask]

def halfkp_collate_fn(batch):
    stm_indices = torch.stack([item['stm_indices'] for item in batch])
    nstm_kings = torch.stack([item['nstm_kings'] for item in batch])
    scores = torch.stack([item['score'] for item in batch])
    wdl = torch.stack([item['wdl'] for item in batch])
    return stm_indices, nstm_kings, (scores, wdl) 

def centipawn_to_prob(score, scale=400.0):
    # Beaucoup plus stable et rapide que 1 / (1 + exp(...))
    return torch.sigmoid(score / scale)


def remove_dicte_keys(state_dict, prefix="_orig_mod."):
    new_state_dict = {}
    for key, value in state_dict.items():
        # Only remove prefix if it exists
        if key.startswith(prefix):
            new_key = key[len(prefix):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

class AlphaScaler():
    def __init__(self):
        super().__init__()
        self.initial_alpha = 0.0
        self.final_alpha = 0.0
        self.total_steps = 0
        self.current_step = 0

    def set_constant_alpha(self, alpha):
        self.initial_alpha = alpha
        self.final_alpha = alpha
        self.total_steps = 1
        self.current_step = 0

    def set_linear_schedule(self, initial_alpha, final_alpha, total_steps):
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1

    def get_alpha(self):
        if self.current_step >= self.total_steps:
            return self.final_alpha
        alpha = self.initial_alpha + (self.final_alpha - self.initial_alpha) * (self.current_step / self.total_steps)
        return alpha
        