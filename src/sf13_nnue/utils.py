import torch
import math
import torch.nn.functional as F
import os

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
    Calcule la perspective adverse (Noirs) à partir de la perspective STM (Blancs).
    Garde le padding (-1) intact.
    """
    # 1. On isole la partie "Pièce + Case" (0-639)
    # Si indices_white est -1, remainder sera -1 (ou 639 selon l'implémentation, 
    # d'où l'importance du masque à la fin).
    remainder = indices_white % 640
    
    # 2. Extraire PieceOffset (0-9) et PieceSq (0-63)
    p_idx_w = torch.div(remainder, 64, rounding_mode='floor')
    p_sq_w = remainder % 64
    
    # 3. Transformer pour la perspective Noire
    p_idx_b = (p_idx_w + 5) % 10 # Swap 0-4 <-> 5-9
    p_sq_b = p_sq_w ^ 56         # Flip vertical de la case
    
    # 4. Case du Roi Noir vue par le Noir (Flip vertical)
    if not isinstance(black_king_sq, torch.Tensor):
        black_king_sq = torch.tensor(black_king_sq, device=indices_white.device)
    k_sq_b_view = black_king_sq ^ 56

    if k_sq_b_view.dim() == 1 and indices_white.dim() == 2:
        k_sq_b_view = k_sq_b_view.unsqueeze(-1)
    
    # 5. Reconstruire l'indice final
    indices_black = (k_sq_b_view * 640) + (p_idx_b * 64) + p_sq_b
    
    # --- SÉCURITÉ CRITIQUE ---
    # Si l'indice d'origine était du padding (-1), on force le résultat à -1
    # Cela évite que le calcul (k_sq * 640 + ...) ne crée une pièce imaginaire.
    indices_black = torch.where(indices_white == -1, -1, indices_black)
    
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

def cp_to_wdl(cp):
    # On ramène souvent le cp à une échelle où 400 = gain quasi certain
    # Le facteur 400 peut être ajusté selon ton "scaling"
    return torch.sigmoid(cp / (400.0 / 4.394)) # 4.394 est une constante de logit

def float_sigmoid(x):
    return 1 / (1 + math.exp(-x))

def cp_to_wdl_float(cp):
    # On ramène souvent le cp à une échelle où 400 = gain quasi certain
    # Le facteur 400 peut être ajusté selon ton "scaling"
    return float_sigmoid(cp / (400.0 / 4.394)) # 4.394 est une constante de logit


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
        
        
        

def save_checkpoint(model, optimizer, scheduler, alpha_scaler, epoch, path):
    """
    Sauvegarde le modèle en nettoyant les préfixes de torch.compile 
    et en incluant les états de l'entraînement.
    """
    # 1. Récupérer le state_dict
    raw_state_dict = model.state_dict()
    
    # 2. Nettoyer les clés (supprimer '_orig_mod.')
    # C'est CRUCIAL pour que serialize.py reconnaisse 'input.weight', etc.
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in raw_state_dict.items()}
    
    # 3. Préparer l'objet complet (pour le resume)
    checkpoint = {
        'epoch': epoch,
        'state_dict': clean_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'alpha_scaler_state': alpha_scaler.state_dict() if hasattr(alpha_scaler, 'state_dict') else alpha_scaler,
    }
    
    # Créer le dossier si nécessaire
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Sauvegarde
    torch.save(checkpoint, path)
    
    # Optionnel : Sauvegarder un fichier "weights only" 
    # Plus facile à pointer pour le script serialize.py
    weights_path = path.replace('.pt', '_weights.pt')
    torch.save(clean_state_dict, weights_path)
    
    print(f"Checkpoint sauvé : {path}")
    
    
def fix_indices_on_the_fly(indices, stm_is_black):
    """
    Si c'est aux noirs de jouer (stm_is_black), on corrige 
    l'index pour passer du Flip Vertical à la Rotation 180.
    """
    if not stm_is_black:
        return indices # Déjà correct pour les blancs
    
    # On décompose l'index (sur CPU ou GPU avec PyTorch)
    ksq_vflip = indices // 640
    remainder = indices % 640
    p_idx = remainder // 64
    psq_vflip = remainder % 64
    
    # On applique le miroir horizontal (^ 7) pour transformer le flip en rotation
    ksq_rot = ksq_vflip ^ 7
    psq_rot = psq_vflip ^ 7
    
    return (ksq_rot * 640) + (p_idx * 64) + psq_rot