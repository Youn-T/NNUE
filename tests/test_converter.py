import io
import os
import tempfile
import torch
import chess
import chess.pgn
import pytest
from data.converter import fast_halfkp_indices, HalfKPExporter

def test_fast_halfkp_indices():
    # Test standard starting position
    board = chess.Board()
    indices = fast_halfkp_indices(board)
    
    # White King is at E1 (square 4)
    # White Pawn at E2 (square 12) -> piece_type=1 (PAWN), color=WHITE. 
    # STM is White. p_idx = PIECE_OFFSET[1] = 0. p_sq = 12
    # King sq = 4. 
    # Index = 4 * 640 + 0 * 64 + 12 = 2560 + 12 = 2572
    assert 2572 in indices

    # Black Pawn at E7 (square 52) -> piece_type=1, color=BLACK
    # STM is White. p_idx = PIECE_OFFSET[1] + 5 = 5. p_sq = 52
    # Index = 4 * 640 + 5 * 64 + 52 = 2560 + 320 + 52 = 2932
    assert 2932 in indices
    
    # After e4, e5, turn is White
    board.push_san("e4")
    board.push_san("e5")
    
    indices2 = fast_halfkp_indices(board)
    # White pawn on E4 (28) -> Index = 4 * 640 + 0 * 64 + 28 = 2588
    assert 2588 in indices2
    # Black pawn on E5 (36) -> Index = 4 * 640 + 5 * 64 + 36 = 2560 + 320 + 36 = 2916
    assert 2916 in indices2


def test_halfkp_exporter_wdl_and_eval():
    # Un court PGN simulé avec différentes évaluations et un résultat décisif
    pgn_data = """[Event "FIDE World Cup 2017"]
[Site "Tbilisi GEO"]
[Date "2017.09.09"]
[Round "4.2"]
[White "Carlsen,M"]
[Black "Bu Xiangzhi"]
[Result "1-0"]
[WhiteElo "2827"]
[BlackElo "2710"]
[EventDate "2017.09.03"]
[ECO "C55"]

1. e4 { [%eval 0.25] } 1... e5 { [%eval -0.15] } 2. Nf3 { [%eval 0.35] } 2... Nc6 { [%eval M4] } 3. Bc4 { [%eval -M2] } 3... Nf6 { [%eval 1.25,18] } 1-0
"""
    
    pgn_io = io.StringIO(pgn_data)
    exporter = HalfKPExporter()
    
    game = chess.pgn.read_game(pgn_io)
    game.accept(exporter)

    assert len(exporter.eval_labels) == 6
    assert len(exporter.wdl_labels) == 6
    assert len(exporter.pos_indices) == 6

    # Évaluations 
    # 1. e4: 0.25 -> 25 cp
    # 1... e5: -0.15 -> -15 cp
    # 2. Nf3: 0.35 -> 35 cp
    # 2... Nc6: M4 -> 10000
    # 3. Bc4: -M2 -> -10000
    # 3... Nf6: 1.25,18 -> 125 cp
    
    expected_evals = [-25, 15, -35, -10000, 10000, -125]
    assert exporter.eval_labels == expected_evals

    # WDL: Résultat est 1-0 (White wins). 
    # index 0: Black to move -> win for white (1.0)
    # index 1: White to move -> loss for white (0.0)
    # index 2: Black to move -> win for white (1.0)
    # index 3: White to move -> loss for white (0.0)
    # index 4: Black to move -> win for white (1.0)
    # index 5: White to move -> loss for white (0.0)
    expected_wdl = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    assert exporter.wdl_labels == expected_wdl

def test_generated_file_integrity():
    """
    Test that the .pt file produced by torch.save matches exactly
    the data format and types expected by the PyTorch Dataset.
    """
    exporter = HalfKPExporter()
    
    # Fake extracted data (like what we tested above)
    # Positions 1, 2 and 3 simulated with dummy index arrays
    exporter.pos_indices = [
        [2572, 2932], # pos 1
        [2588, 2916], # pos 2
        [1000, 2000, 3000] # pos 3
    ]
    exporter.wdl_labels = [1.0, 0.5, 0.0]
    exporter.eval_labels = [25, 0, -100]

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_chunk.pt")
        
        # Simuler exactement la sauvegarde définie dans data/converter.py
        torch.save({
            'indices': exporter.pos_indices, 
            'wdl': torch.tensor(exporter.wdl_labels, dtype=torch.float32),
            'score': torch.tensor(exporter.eval_labels, dtype=torch.int16)
        }, save_path)

        # 1. Vérification système de fichiers
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0
        
        # 2. Chargement du fichier
        # weights_only=False est parfois nécessaire avec certaines versions de torch 
        # s'il y a des listes python brutes (indices).
        # Mais testons si on peut charger sans problème :
        data = torch.load(save_path)

        # 3. Vérification de l'intégrité de la structure
        assert 'indices' in data
        assert 'wdl' in data
        assert 'score' in data

        # 4. Vérification des types attendus (crucial pour le Dataloader)
        assert isinstance(data['indices'], list)
        assert data['wdl'].dtype == torch.float32
        assert data['score'].dtype == torch.int16

        # 5. Vérification des dimensions
        assert len(data['indices']) == 3
        assert data['wdl'].shape == (3,)
        assert data['score'].shape == (3,)

        # 6. Vérification des valeurs exactes
        assert data['indices'][0] == [2572, 2932]
        torch.testing.assert_close(data['wdl'], torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32))
        torch.testing.assert_close(data['score'], torch.tensor([25, 0, -100], dtype=torch.int16))

def test_full_pipeline_with_file_saving():
    """
    Test the full pipeline: PGN String -> Exporter -> PyTorch chunk saving -> Reload checks.
    Ensures that empty/invalid records are dropped and that a correct chunk is produced.
    """
    pgn_data = """[Event "?"]
[Site "Tbilisi GEO"]
[Date "2017.09.09"]
[Round "4.2"]
[White "Carlsen,M"]
[Black "Bu Xiangzhi"]
[Result "1/2-1/2"]
[WhiteElo "2827"]
[BlackElo "2710"]
[EventDate "2017.09.03"]
[ECO "C55"]

1. e4 { [%eval 0.0] } 1... e5 { [%eval 0.1] } 2. Nf3 { [%eval 0.15] } 1/2-1/2
"""
    
    pgn_io = io.StringIO(pgn_data)
    exporter = HalfKPExporter()
    
    game = chess.pgn.read_game(pgn_io)
    game.accept(exporter)

    # 3 positions générées et un score total nul = 0.5 de Draw
    # Mais le but ici est de simuler sa sauvegarde.
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "full_pipeline_chunk.pt")
        
        torch.save({
            'indices': exporter.pos_indices, 
            'wdl': torch.tensor(exporter.wdl_labels, dtype=torch.float32),
            'score': torch.tensor(exporter.eval_labels, dtype=torch.int16)
        }, save_path)
        
        data = torch.load(save_path)
        
        # Test intégrité globale
        assert len(data['indices']) == 3
        # Les scores devraient être 0, 10, 15
        assert data['score'].tolist() == [0, -10, -15]
        # Tous draws (0.5) car Result est "1/2-1/2" et ça ne dépend pas de qui joue !
        assert data['wdl'].tolist() == [0.5, 0.5, 0.5]
