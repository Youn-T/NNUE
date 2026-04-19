import os, glob, gzip, chess, chess.pgn, torch, multiprocessing as mp
import numpy as np
import re

# --- CONFIGURATION ---
INPUT_DIR = "data/raw_dataset"
OUTPUT_DIR = "data/halfkp_data"
CHUNK_SIZE = 1_000_000  # On augmente la taille des chunks pour moins de fichiers

PIECE_OFFSET = [0, 0, 1, 2, 3, 4] 

def fast_halfkp_indices(board):
    """Calcule les indices sans aucune allocation d'objet complexe."""
    stm = board.turn
    king_sq = board.king(stm)
    if stm == chess.BLACK:
        king_sq ^= 56 

    indices = []
    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING: continue
        p_idx = PIECE_OFFSET[piece.piece_type]
        if piece.color != stm: p_idx += 5
        p_sq = sq if stm == chess.WHITE else sq ^ 56
        indices.append(king_sq * 640 + p_idx * 64 + p_sq)
    return indices

class HalfKPExporter(chess.pgn.BaseVisitor):
    def __init__(self):
        self.pos_indices = []
        self.wdl_labels = []
        self.eval_labels = []
        self.stm_kings = []
        self.nstm_kings = []
        
        self.res_val = 0.5
        self.board = None # Référence vers le plateau interne
        self.eval_re = re.compile(r"\[%eval\s+([^\]]+)\]|^([+-]?(?:\d+\.\d+|M\d+|M-[0-9]+|\#[-+]?\d+|\d+))(?:/|\s|$)")

    def begin_game(self):
        self.res_val = 0.5
        return None 

    def visit_header(self, name, value):
        if name == "Result":
            if value == "1-0": self.res_val = 1.0
            elif value == "0-1": self.res_val = 0.0
            else: self.res_val = 0.5

    def visit_board(self, board):
        # On garde une référence vers le plateau que le parseur va manipuler
        self.board = board

    def visit_move(self, board, move):
        # On ne fait rien ici, on attend le commentaire qui suit le coup
        pass

    def visit_comment(self, comment):
        """
        Déclencheur principal : ici, le coup a déjà été joué sur self.board.
        On extrait l'éval et on enregistre l'état actuel du plateau.
        """
        match = self.eval_re.search(comment)
        if match:
            # 1. Extraction du score (toujours relatif aux blancs dans le PGN)
            val = match.group(1) or match.group(2)
            val = val.split(",", 1)[0].strip()
            
            if 'M' in val or '#' in val:
                val_int = int(val.replace('M', '').replace('#', ''))
                score_white = 10000 if val_int > 0 else -10000
            else:
                score_white = int(float(val) * 100)
            
            # CORRECTION : Selon tes tests, le label attendu est -score_white
            # car après le coup, c'est au tour de l'adversaire (STM).
            self.eval_labels.append(-score_white)

            # 2. Indices HalfKP (pour le joueur dont c'est le tour MAINTENANT)
            self.pos_indices.append(fast_halfkp_indices(self.board))

            # 3. WDL (Perspective Side To Move)
            wdl = self.res_val if self.board.turn == chess.BLACK else 1.0 - self.res_val
            self.wdl_labels.append(wdl)

            # 4. Rois (Perspective STM)
            if self.board.turn == chess.WHITE:
                stm_k = self.board.king(chess.WHITE)
                nstm_k = self.board.king(chess.BLACK)
            else:
                stm_k = self.board.king(chess.BLACK) ^ 56
                nstm_k = self.board.king(chess.WHITE) ^ 56
            
            self.stm_kings.append(stm_k)
            self.nstm_kings.append(nstm_k)

    def result(self):
        return True

def process_file_chunk(args):
    worker_id, file_paths = args
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    exporter = HalfKPExporter()
    chunk_idx = 0
    
    print(f"[W{worker_id}] Start.", flush=True)

    for file_path in file_paths:
        try:
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                game_counter = 0
                while True:
                    # L'astuce magique : on passe l'exporter à read_game
                    # Ça ne renvoie pas de "game", ça remplit exporter.pos_indices
                    # read_game attend un callable (factory) pour Visitor.
                    result = chess.pgn.read_game(f, Visitor=lambda: exporter)
                    if result is None: break
                    
                    game_counter += 1
                    
                    if game_counter % 100 == 0:
                        print(f"[Worker {worker_id}] Fichier {os.path.basename(file_path)} : {game_counter} parties lues...", flush=True)

                    
                    # Sauvegarde si on dépasse le CHUNK_SIZE
                    if len(exporter.pos_indices) >= CHUNK_SIZE:
                        save_path = os.path.join(OUTPUT_DIR, f"w{worker_id}_{chunk_idx}_sz{len(exporter.pos_indices)}.pt")
                        # # Conversion massive en une seule fois
                        # torch.save({
                        #     'indices': exporter.pos_indices, 
                        #     'labels': torch.tensor(exporter.labels, dtype=torch.float32)
                        # }, save_path)
                        
                        max_pieces = 30 
                        padded_indices = torch.full((len(exporter.pos_indices), max_pieces), -1, dtype=torch.int32)

                        for i, idx_list in enumerate(exporter.pos_indices):
                            l = len(idx_list)
                            padded_indices[i, :l] = torch.tensor(idx_list, dtype=torch.int32)
                        
                        torch.save({
                            'indices': padded_indices,
                            'wdl': torch.tensor(exporter.wdl_labels, dtype=torch.float32),
                            'score': torch.tensor(exporter.eval_labels, dtype=torch.int16), # Centipawns tiennent en int16
                            'stm_kings': torch.tensor(exporter.stm_kings, dtype=torch.int16),
                            'nstm_kings': torch.tensor(exporter.nstm_kings, dtype=torch.int16)
                        }, save_path)
                        
                        print(f"✅ [W{worker_id}] Saved {chunk_idx}", flush=True)
                        exporter.pos_indices = []
                        exporter.wdl_labels = []
                        exporter.eval_labels = []
                        exporter.stm_kings = []
                        exporter.nstm_kings = []
                        chunk_idx += 1
                        
        except Exception as e:
            print(f"Error W{worker_id}: {e}", flush=True)

if __name__ == '__main__':
    mp.freeze_support()
    # 1. Lister tous les fichiers .pgn.gz
    print("Recherche des fichiers...")
    search_pattern = os.path.join(INPUT_DIR, "**", "*.pgn.gz")
    all_files = glob.glob(search_pattern, recursive=True)
    print(f"{len(all_files)} fichiers trouvés.")
    
    if len(all_files) == 0:
        print("Vérifiez le chemin INPUT_DIR.")
        exit()

    # 2. Répartir la charge sur les cœurs CPU
    num_cores = max(1, mp.cpu_count() - 1) # Garde 1 cœur pour le système
    print(f"Lancement du multiprocessing sur {num_cores} cœurs...")
    
    # Répartition gloutonne par taille de fichier pour réduire les fins de traitement "bloquées".
    file_sizes = []
    for path in all_files:
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        file_sizes.append((path, size))

    file_sizes.sort(key=lambda x: x[1], reverse=True)
    chunked_files = [[] for _ in range(num_cores)]
    worker_loads = [0 for _ in range(num_cores)]

    for path, size in file_sizes:
        target = min(range(num_cores), key=lambda i: worker_loads[i])
        chunked_files[target].append(path)
        worker_loads[target] += size
    
    # Création des arguments pour chaque worker : (ID_worker, Liste_de_fichiers)
    worker_args = [(i, chunk) for i, chunk in enumerate(chunked_files)]
    
    print(f"Chaque worker traitera environ {len(all_files) // num_cores} fichiers.")
    print(
        "Charge estimee par worker (Go): "
        + ", ".join(f"w{i}={load / (1024**3):.2f}" for i, load in enumerate(worker_loads))
    )
    # 3. Lancement !
    with mp.Pool(num_cores) as pool:
        pool.map(process_file_chunk, worker_args)
        
    print(f"\nConversion terminée ! Les fichiers .pt sont dans {OUTPUT_DIR}")