import os, json, chess, torch, multiprocessing as mp
from sf13_nnue.utils import cp_to_wdl_float

# --- CONFIGURATION ---
INPUT_FILE = "D:/Projects/Lichess Dataset/lichess_db_eval.jsonl/lichess_db_eval.jsonl"
OUTPUT_DIR = "D:/Projects/HalfKP Dataset Lichess 2"
CHUNK_SIZE = 1_000_000 

PIECE_OFFSET = [0, 0, 1, 2, 3, 4] 

def fast_halfkp_indices(board):
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

def save_tensors(worker_id, chunk_idx, pos_indices, eval_labels, wdl_labels, stm_kings, nstm_kings):
    save_path = os.path.join(OUTPUT_DIR, f"w{worker_id}_{chunk_idx}_sz{len(pos_indices)}.pt")
    max_pieces = 30 
    padded_indices = torch.full((len(pos_indices), max_pieces), -1, dtype=torch.int32)

    for i, idx_list in enumerate(pos_indices):
        l = len(idx_list)
        padded_indices[i, :l] = torch.tensor(idx_list, dtype=torch.int32)
    
    torch.save({
        'indices': padded_indices,
        'wdl': torch.tensor(wdl_labels, dtype=torch.float32),
        'score': torch.tensor(eval_labels, dtype=torch.int16),
        'stm_kings': torch.tensor(stm_kings, dtype=torch.int16),
        'nstm_kings': torch.tensor(nstm_kings, dtype=torch.int16)
    }, save_path)
    
    print(f"✅ [W{worker_id}] Saved {chunk_idx}", flush=True)

def process_byte_chunk(args):
    worker_id, start_byte, end_byte, file_path = args
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    pos_indices, eval_labels, wdl_labels, stm_kings, nstm_kings = [], [], [], [], []
    chunk_idx = 0
    
    with open(file_path, 'rb') as f:
        f.seek(start_byte)
        
        while f.tell() < end_byte:
            line = f.readline()
            if not line: break
            
            try:
                data = json.loads(line.decode('utf-8'))
                evals = data.get("evals", [])
                if not evals: continue
                
                # 1. Récupération de l'éval avec la plus haute depth
                best_eval = max(evals, key=lambda x: x.get("depth", 0))
                pvs = best_eval.get("pvs", [])
                if not pvs: continue
                
                # 2. Récupération du premier pv et skip des mats
                first_pv = pvs[0]
                if "mate" in first_pv: continue
                
                board = chess.Board(data["fen"])
                cp_abs = first_pv["cp"]
                
                if len(pos_indices) % 2 == 1: 
                    board.turn = chess.WHITE if board.turn == chess.BLACK else chess.BLACK # Inverser le trait pour augmenter la diversité des positions
                    cp_abs = -cp_abs # Inverser aussi l'éval pour rester cohérent avec le trait
                
                stm = board.turn
                
                # 3. Conversion de l'éval absolue (blancs) -> relative (trait)
                score_stm = cp_abs if stm == chess.WHITE else -cp_abs
                indices = fast_halfkp_indices(board)
                if (len(indices) > 30): 
                    print(board.fen(), len(indices))
                    continue # Skip les positions trop complexes pour notre modèle actuel
                pos_indices.append(indices)
                eval_labels.append(score_stm)
                wdl_labels.append(cp_to_wdl_float(score_stm)) # WDL basé sur l'éval absolue
                
                if stm == chess.WHITE:
                    stm_kings.append(board.king(chess.WHITE))
                    nstm_kings.append(board.king(chess.BLACK))
                else:
                    stm_kings.append(board.king(chess.BLACK) ^ 56)
                    nstm_kings.append(board.king(chess.WHITE) ^ 56)

            except Exception as e:
                print(f"[W{worker_id}] Error processing line: {e}")
                continue # Ignore les lignes corrompues
            if len(pos_indices) % 1000 == 0:
                print(f"[W{worker_id}] Processed {len(pos_indices)} positions...", flush=True)
            
            if len(pos_indices) >= CHUNK_SIZE:
                save_tensors(worker_id, chunk_idx, pos_indices, eval_labels, wdl_labels, stm_kings, nstm_kings)
                pos_indices, eval_labels, wdl_labels, stm_kings, nstm_kings = [], [], [], [], []
                chunk_idx += 1

        # Sauvegarde du reliquat à la fin du chunk
        if len(pos_indices) > 0:
            save_tensors(worker_id, chunk_idx, pos_indices, eval_labels, wdl_labels, stm_kings, nstm_kings)

if __name__ == '__main__':
    mp.freeze_support()
    
    if not os.path.exists(INPUT_FILE):
        print(f"Fichier {INPUT_FILE} introuvable.")
        exit()

    num_cores = max(1, mp.cpu_count() - 1)
    file_size = os.path.getsize(INPUT_FILE)
    chunk_size = file_size // num_cores
    
    # Stratégie : Diviser le fichier géant en segments d'octets.
    # Chaque processus commencera à lire à un offset précis jusqu'à son marqueur de fin.
    chunks = []
    with open(INPUT_FILE, 'rb') as f:
        start = 0
        for i in range(num_cores):
            f.seek(start + chunk_size)
            f.readline() # Avance jusqu'au prochain saut de ligne pour ne pas couper un JSON en deux
            end = f.tell()
            if i == num_cores - 1: end = file_size
            chunks.append((i, start, end, INPUT_FILE))
            start = end
            
    print(f"Lancement sur {num_cores} cœurs (lecture RAM directe)...")
    with mp.Pool(num_cores) as pool:
        pool.map(process_byte_chunk, chunks)
        
    print(f"\nConversion terminée ! Les fichiers .pt sont dans {OUTPUT_DIR}")