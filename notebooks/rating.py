
import multiprocessing as mp


import chess
import torch
import torch.nn as nn
import time

from stockfish import Stockfish

# ==========================================
# 1. DÉFINITION DU MODÈLE PYTORCH
# ==========================================
class CReLU(nn.Module):
    def __init__(self, clip_value=255.0):
        super().__init__()
        self.clip_value = clip_value

    def forward(self, x):
        return torch.clamp(x, min=0.0, max=self.clip_value)

class ChessNNUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_transformer = nn.Embedding(40961, 768)
        
        # On remplace tous les nn.ReLU() par CReLU(255.0)
        self.linear_stack = nn.Sequential(
            nn.Linear(1536, 512),
            CReLU(),
            nn.Linear(512, 32),
            CReLU(),
            nn.Linear(32, 32),
            CReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, us_indices, them_indices):
        # 1. On somme les poids (Accumulateur)
        us_acc = self.feature_transformer(us_indices).sum(dim=1)
        them_acc = self.feature_transformer(them_indices).sum(dim=1)
        
        # 2. TRÈS IMPORTANT : Le clamp de l'accumulateur (CReLU)
        us_acc = torch.clamp(us_acc, 0.0, 255.0)
        them_acc = torch.clamp(them_acc, 0.0, 255.0)
        
        combined = torch.cat([us_acc, them_acc], dim=1)
        out = self.linear_stack(combined)
        return out

# ==========================================
# 2. EXTRACTION DES FEATURES (HalfKP)
# ==========================================
def get_halfkp_indices(board: chess.Board, pov_color: chess.Color):
    """
    Traduit l'échiquier en indices HalfKP selon la perspective d'une couleur.
    """
    king_sq = board.king(pov_color)
    if pov_color == chess.BLACK:
        king_sq ^= 56 # Flip board pour le roi noir

    indices = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        # On ignore les cases vides et les Rois (qui sont déjà encodés par king_sq)
        if piece is None or piece.piece_type == chess.KING:
            continue
            
        # Calcul de l'index de la pièce (0-4 pour nous, 5-9 pour l'adversaire)
        # python-chess: PAWN=1, KNIGHT=2, BISHOP=3, ROOK=4, QUEEN=5
        pc_idx = piece.piece_type - 1 
        if piece.color != pov_color:
            pc_idx += 5
            
        # Orientation de la case
        p_sq = sq
        if pov_color == chess.BLACK:
            p_sq ^= 56
            
        # Formule HalfKP classique : (Roi * 640) + (Piece * 64) + Case
        idx = king_sq * 640 + pc_idx * 64 + p_sq
        indices.append(idx)
        
    return indices

def evaluate_board(board: chess.Board, model: nn.Module) -> float:
    """
    Évalue la position du point de vue du joueur qui doit jouer.
    """
    # Récupération des indices pour les deux camps
    us_color = board.turn
    them_color = not us_color
    
    us_indices = get_halfkp_indices(board, us_color)
    them_indices = get_halfkp_indices(board, them_color)
    
    # Conversion en tenseurs (batch_size = 1)
    us_tensor = torch.tensor([us_indices], dtype=torch.long)
    them_tensor = torch.tensor([them_indices], dtype=torch.long)
    
    # Mode évaluation (désactive les gradients pour la vitesse)
    model.eval()
    with torch.no_grad():
        score = model(us_tensor, them_tensor)
        
    return score.item()

# ==========================================
# 3. RECHERCHE ET TRI DE COUPS
# ==========================================
def order_moves(board: chess.Board):
    """Trie les coups possibles pour optimiser l'élagage alpha-beta (captures en premier)."""
    moves = list(board.legal_moves)
    moves.sort(key=lambda m: board.is_capture(m), reverse=True)
    return moves

def quiesce(board: chess.Board, model: nn.Module, alpha: float, beta: float, depth: int) -> float:
    """
    Recherche de quiescence: évaluer seulement les captures 
    jusqu'à une position calme pour éviter l'effet d'horizon.
    """
    stand_pat = evaluate_board(board, model)
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat
        
    for move in order_moves(board):
        if depth <= 0:
            break
        if board.is_capture(move):
            board.push(move)
            score = -quiesce(board, model, -beta, -alpha, depth - 1)
            board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
    return alpha

def negamax(board: chess.Board, depth: int, model: nn.Module, alpha: float, beta: float) -> float:
    if depth == 0:
        return quiesce(board, model, alpha, beta, depth=10)
        
    if board.is_game_over():
        if board.is_checkmate():
            return -100000.0 # Pire score si on est mat
        return 0.0 # Nul (pat, répétition, etc.)

    best_value = -float('inf')
    
    for move in order_moves(board):
        board.push(move)
        score = -negamax(board, depth - 1, model, -beta, -alpha)
        board.pop()
        
        best_value = max(best_value, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break # Coupure alpha-beta
            
    return best_value

def get_best_move(board: chess.Board, depth: int, model: nn.Module) -> chess.Move:
    best_move = None
    best_value = -float('inf')
    alpha = -float('inf')
    beta = float('inf')
    
    for move in order_moves(board):
        board.push(move)
        score = -negamax(board, depth - 1, model, -beta, -alpha)
        board.pop()
        
        if score > best_value:
            best_value = score
            best_move = move
            
        alpha = max(alpha, score)
        
    # print(f"Meilleur coup trouvé : {best_move} (Score: {best_value:.3f})")
    return best_move

# ==========================================
# 4. SCRIPT PRINCIPAL
# ==========================================



def game(args):
    worker_id, elo = args
    
    print(f"Worker {worker_id} processing ELO {elo}...")
    
    stockfish = Stockfish(path="C:/Users/yount/Downloads/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe")
    stockfish.set_elo_rating(elo)
    
    print("Initialisation du modèle...")
    model = ChessNNUE()
    
    # 2. Charger les poids de manière robuste (gestion du '_orig_mod.')
    model_path = "weights/weights4_tuned4/model_weights_4.pth" # REMPLACE PAR LE NOM DE TON FICHIER
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        clean_state_dict = {}
        for key, value in state_dict.items():
            clean_key = key.replace("_orig_mod.", "")
            clean_state_dict[clean_key] = value
            
        model.load_state_dict(clean_state_dict)
        print("Poids chargés avec succès !")
    except Exception as e:
        print(f"Attention, erreur lors du chargement des poids : {e}")
        print("Le script va continuer avec des poids aléatoires pour le test.")
    
    board = chess.Board()
    while not board.is_game_over():
            
        best_move = get_best_move(board, depth=3, model=model) # Profondeur 3 + Quiescence
        
        
        if best_move:
            board.push(best_move)
            
        stockfish.set_fen_position(board.fen())
        sf_move = stockfish.get_best_move()
        board.push(chess.Move.from_uci(sf_move))
    print(f"[{worker_id}]: Elo: {elo}. Game over. Result: {board.result()}")    

if __name__ == "__main__":
    mp.freeze_support()
    num_cores = 5#max(1, mp.cpu_count() - 1)
    worker_args = []
    for i in range(num_cores):
        worker_args.append((i, 1000 + i*100))
        worker_args.append((i, 1500 + i*100))
        # print(f"Worker {i} will process ELO {2000 + i*100} to {2099 + i*100}")

    with mp.Pool(num_cores) as pool:
        pool.map(game, worker_args)