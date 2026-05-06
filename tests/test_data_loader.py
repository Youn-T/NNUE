import torch
from torch.utils.data import DataLoader

from sf13_nnue.data_loader import HalfKPDataset
from sf13_nnue.utils import get_nstm_indices, halfkp_collate_fn


def _save_chunk(path, indices, wdl, score, stm_kings, nstm_kings):
    torch.save(
        {
            "indices": torch.tensor(indices, dtype=torch.int32),
            "wdl": torch.tensor(wdl, dtype=torch.float32),
            "score": torch.tensor(score, dtype=torch.int16),
            "stm_kings": torch.tensor(stm_kings, dtype=torch.int16),
            "nstm_kings": torch.tensor(nstm_kings, dtype=torch.int16),
        },
        path,
    )


def test_halfkp_dataset_len_and_file_accounting(tmp_path):
    _save_chunk(
        tmp_path / "w0_0_sz2.pt",
        indices=[[2572, 2932], [2588, 2916]],
        wdl=[1.0, 0.0],
        score=[25, -30],
        stm_kings=[4, 4],
        nstm_kings=[60, 60],
    )
    _save_chunk(
        tmp_path / "w0_1_sz3.pt",
        indices=[[100, 200], [300, 400], [500, 600]],
        wdl=[0.5, 1.0, 0.0],
        score=[0, 15, -40],
        stm_kings=[4, 4, 4],
        nstm_kings=[60, 60, 60],
    )

    dataset = HalfKPDataset(data_dir=str(tmp_path))

    assert len(dataset) == 5
    assert dataset.samples_per_file == [2, 3]
    assert dataset.cumulative_sizes.tolist() == [2, 5]


def test_halfkp_dataset_getitem_cross_file_and_types(tmp_path):
    _save_chunk(
        tmp_path / "w0_0_sz1.pt",
        indices=[[2572, 2932]],
        wdl=[1.0],
        score=[25],
        stm_kings=[4],
        nstm_kings=[60],
    )
    _save_chunk(
        tmp_path / "w0_1_sz1.pt",
        indices=[[2588, 2916]],
        wdl=[0.0],
        score=[-15],
        stm_kings=[4],
        nstm_kings=[60],
    )

    dataset = HalfKPDataset(data_dir=str(tmp_path))

    item0 = dataset[0]
    expected0 = get_nstm_indices(torch.tensor([2572, 2932], dtype=torch.long), 60)

    assert torch.equal(item0["stm_indices"], torch.tensor([2572, 2932], dtype=torch.long))
    assert torch.equal(item0["nstm_indices"], expected0)
    assert item0["stm_indices"].dtype == torch.long
    assert item0["wdl"].dtype == torch.float32
    assert item0["score"].dtype == torch.float32

    item1 = dataset[1]
    expected1 = get_nstm_indices(torch.tensor([2588, 2916], dtype=torch.long), 60)

    assert torch.equal(item1["stm_indices"], torch.tensor([2588, 2916], dtype=torch.long))
    assert torch.equal(item1["nstm_indices"], expected1)


def test_halfkp_dataset_reuses_loaded_file_for_same_chunk(tmp_path):
    _save_chunk(
        tmp_path / "w0_0_sz2.pt",
        indices=[[111, 222], [333, 444]],
        wdl=[1.0, 0.0],
        score=[10, -10],
        stm_kings=[4, 4],
        nstm_kings=[60, 60],
    )

    dataset = HalfKPDataset(data_dir=str(tmp_path))

    _ = dataset[0]
    first_loaded_file = dataset.current_file_idx
    _ = dataset[1]
    second_loaded_file = dataset.current_file_idx

    assert first_loaded_file == 0
    assert second_loaded_file == 0


def test_dataloader_with_halfkp_collate_fn(tmp_path):
    _save_chunk(
        tmp_path / "w0_0_sz2.pt",
        indices=[[2572, 2932], [2588, 2916]],
        wdl=[1.0, 0.0],
        score=[25, -15],
        stm_kings=[4, 4],
        nstm_kings=[60, 60],
    )

    dataset = HalfKPDataset(data_dir=str(tmp_path))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=halfkp_collate_fn)

    (stm_batch, nstm_batch), (score_batch, wdl_batch) = next(iter(dataloader))

    assert stm_batch.shape == (2, 2)
    assert nstm_batch.shape == (2, 2)
    assert score_batch.shape == (2,)
    assert wdl_batch.shape == (2,)

    assert stm_batch.dtype == torch.long
    assert nstm_batch.dtype == torch.long
    assert score_batch.dtype == torch.float32
    assert wdl_batch.dtype == torch.float32


def test_halfkp_dataset_restores_wrapped_int16_indices(tmp_path):
    # 33000 and 40959 overflow if stored in int16 and become negative values.
    wrapped = torch.tensor([[33000, 40959]], dtype=torch.int32).to(torch.int16)

    torch.save(
        {
            "indices": wrapped,
            "wdl": torch.tensor([1.0], dtype=torch.float32),
            "score": torch.tensor([10], dtype=torch.int16),
            "stm_kings": torch.tensor([4], dtype=torch.int16),
            "nstm_kings": torch.tensor([60], dtype=torch.int16),
        },
        tmp_path / "w0_0_sz1.pt",
    )

    dataset = HalfKPDataset(data_dir=str(tmp_path))
    item = dataset[0]

    assert item["stm_indices"].dtype == torch.long
    assert item["stm_indices"].tolist() == [33000, 40959]
