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
    """Visitor ultra-rapide qui extrait les indices au vol."""
    def __init__(self):
        self.pos_indices = []
        self.wdl_labels = []
        self.eval_labels = []
        self.res_val = 0.5
        self.last_eval = 0  # Par défaut si pas d'eval dans le commentaire
        # Regex large: accepte [%eval 0.15], [%eval #2], [%eval 0.15,23], etc.
        # Accepte aussi Fishtest: +0.79/18 0.93s -> prend +0.79
        self.eval_re = re.compile(r"\[%eval\s+([^\]]+)\]|^([+-]?(?:\d+\.\d+|M\d+|M-[0-9]+|\#[-+]?\d+|\d+))(?:/|\s|$)")

    def begin_game(self):
        self.last_eval = 0  # Réinitialiser l'eval pour chaque partie
        return chess.Board()

    def visit_header(self, name, value):
        if name == "Result":
            if value == "1-0": self.res_val = 1.0
            elif value == "0-1": self.res_val = 0.0
            else: self.res_val = 0.5

    def visit_comment(self, comment):
        # Dans Fishtest, l'eval est dans le commentaire du coup
        match = self.eval_re.search(comment)
        if match:
            # Soit group(1) ([%eval]), soit group(2) (Fishtest)
            val = match.group(1) or match.group(2)
            # Certains PGN ajoutent une profondeur: "0.23,18" -> on garde la valeur.
            val = val.split(",", 1)[0].strip()
            
            if 'M' in val:
                # Stockfish format M2 ou -M2 ou +M69
                val = val.replace('M', '#')
                
            if '#' in val: # Cas du mat (#2, #-2, +#69)
                val_int = int(val.replace('#', ''))
                score = 10000 if val_int > 0 else -10000
            else:
                score = int(float(val) * 100) # Conversion en centipawns
            self.last_eval = score

    def visit_move(self, board, move):
        # On stocke AVANT le move
        self.pos_indices.append(fast_halfkp_indices(board))
        self.wdl_labels.append(self.res_val if board.turn == chess.WHITE else 1.0 - self.res_val)
        
        # On ajuste l'eval à la perspective (très important !)
        # Si c'est au tour des noirs, l'eval positive est mauvaise pour eux
        adj_eval = self.last_eval if board.turn == chess.WHITE else -self.last_eval
        self.eval_labels.append(adj_eval)
        
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
                        save_path = os.path.join(OUTPUT_DIR, f"w{worker_id}_{chunk_idx}.pt")
                        # # Conversion massive en une seule fois
                        # torch.save({
                        #     'indices': exporter.pos_indices, 
                        #     'labels': torch.tensor(exporter.labels, dtype=torch.float32)
                        # }, save_path)
                        
                        torch.save({
                            'indices': exporter.pos_indices, 
                            'wdl': torch.tensor(exporter.wdl_labels, dtype=torch.float32),
                            'score': torch.tensor(exporter.eval_labels, dtype=torch.int16) # Centipawns tiennent en int16
                        }, save_path)
                        
                        print(f"✅ [W{worker_id}] Saved {chunk_idx}", flush=True)
                        exporter.pos_indices = []
                        exporter.wdl_labels = []
                        exporter.eval_labels = []
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