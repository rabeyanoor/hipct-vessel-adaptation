"""
Checks of the code that every number in the paper depends on: the metrics,
the bootstrap, the RLE decoder for the challenge predictions, the spacing
encoding, LoRA / FiLM initialisation, the FiLM sweep, the loss, the
post-processing and the sliding-window inference (resampling, flips).

Run from the repository root:  python -m pytest tests -q
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from benchmark.score import bootstrap_ci, score_roi


def cube(shape=(32, 32, 32), lo=(10, 10, 10), size=10):
    m = np.zeros(shape, dtype=bool)
    m[lo[0]:lo[0] + size, lo[1]:lo[1] + size, lo[2]:lo[2] + size] = True
    return m


def tube(shape=(40, 40, 40), r=3, axis=0, center=(20, 20)):
    z, y, x = np.indices(shape)
    a, b = [(y, x), (z, x), (z, y)][axis]
    return (a - center[0]) ** 2 + (b - center[1]) ** 2 <= r * r


# ---------------------------------------------------------------- metrics
def test_identical_masks_are_perfect():
    g = tube()
    r = score_roi(g, g)
    assert r["dice"] == pytest.approx(1.0)
    assert r["cldice"] == pytest.approx(1.0)
    assert r["nsd_t0"] == pytest.approx(1.0)
    assert r["nsd_t1"] == pytest.approx(1.0)
    assert r["hd95"] == pytest.approx(0.0)
    assert r["assd"] == pytest.approx(0.0)
    assert r["pred_components"] == r["gt_components"] == 1


def test_dice_of_shifted_cube():
    # 10^3 cube shifted by one voxel: overlap 9*10*10 -> Dice 2*900/2000
    a, b = cube(), cube(lo=(11, 10, 10))
    r = score_roi(a, b)
    assert r["dice"] == pytest.approx(0.9)
    # a 1-voxel shift is inside the 1-voxel tolerance but not the 0-voxel one
    assert r["nsd_t1"] == pytest.approx(1.0)
    assert r["nsd_t0"] < 1.0
    assert r["hd95"] == pytest.approx(1.0)


def test_disjoint_and_empty_prediction():
    g = cube()
    r = score_roi(cube(lo=(0, 0, 0), size=5), g)
    assert r["dice"] == 0.0 and r["cldice"] == 0.0
    r = score_roi(np.zeros_like(g), g)
    assert r["dice"] == 0.0 and r["nsd_t0"] == 0.0 and np.isnan(r["hd95"])


def test_scores_do_not_depend_on_padding():
    # score_roi crops to the union of the masks; padding must not change anything
    p, g = tube(r=3), tube(r=4, center=(21, 20))
    r1 = score_roi(p, g)
    r2 = score_roi(np.pad(p, 15), np.pad(g, 15))
    for k in r1:
        assert r1[k] == pytest.approx(r2[k], abs=1e-6), k


def test_mask_touching_the_border():
    # vessels that leave the ROI: the crop must keep the ROI border as is
    p, g = tube(r=3), tube(r=3)
    g[:, 18:22, 18:22] = True
    r1 = score_roi(p, g)
    r2 = score_roi(np.pad(p, 5), np.pad(g, 5))
    assert r1["dice"] == pytest.approx(r2["dice"])
    assert r1["cldice"] == pytest.approx(r2["cldice"])


def test_cldice_rewards_connectivity():
    g = tube(r=3)
    broken = g.copy()
    broken[18:22] = False  # cut the vessel
    full = score_roi(g, g)["cldice"]
    cut = score_roi(broken, g)
    assert cut["cldice"] < full
    assert cut["pred_components"] == 2


def test_bootstrap_ci():
    m, lo, hi = bootstrap_ci([0.5] * 8)
    assert m == lo == hi == 0.5
    m, lo, hi = bootstrap_ci([0.0, 1.0, float("nan"), 0.5])
    assert m == pytest.approx(0.5) and lo <= m <= hi
    assert bootstrap_ci([0.1, 0.2, 0.3], seed=0) == bootstrap_ci([0.1, 0.2, 0.3], seed=0)


# ---------------------------------------------------------------- RLE
def test_rle_decode_row_major_one_based():
    from benchmark.import_team_rle import rle_decode
    m = rle_decode("1 3 6 2", (2, 4))
    assert m.tolist() == [[1, 1, 1, 0], [0, 1, 1, 0]]
    assert rle_decode("", (3, 3)).sum() == 0


def test_rle_round_trip():
    from benchmark.import_team_rle import rle_decode
    rng = np.random.default_rng(0)
    m = (rng.random((17, 23)) > 0.7).astype(np.uint8)
    # the challenge's encoder (row-major, 1-based starts)
    pixels = np.concatenate([[0], m.flatten(), [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    rle = " ".join(map(str, runs))
    assert np.array_equal(rle_decode(rle, m.shape), m)


# ---------------------------------------------------------------- spacing, LoRA, FiLM
def test_log_spacing_encoding():
    from models.spacing_lora import encode_spacing
    s = torch.tensor([[50.0, 25.0, 100.0]])
    assert torch.allclose(encode_spacing(s, "log"), torch.tensor([[0.0, -1.0, 1.0]]))
    assert torch.equal(encode_spacing(s, "raw"), s)
    with pytest.raises(ValueError):
        encode_spacing(s, "other")


def test_lora_starts_as_identity():
    from models.lora import LoRALinear
    torch.manual_seed(0)
    base = nn.Linear(12, 20)
    x = torch.randn(2, 5, 12)
    ref = base(x)
    layer = LoRALinear(base, r=4)
    assert torch.allclose(layer(x), ref)
    assert not any(p.requires_grad for p in layer.base.parameters())


@pytest.mark.parametrize("encoding", ["raw", "log"])
def test_spacing_lora_starts_as_identity(encoding):
    from models.spacing_lora import SpacingConditionedLoRALinear
    torch.manual_seed(0)
    base = nn.Linear(12, 20)
    x = torch.randn(2, 5, 12)
    layer = SpacingConditionedLoRALinear(base, r=4, spacing_encoding=encoding)
    for s in (5.2, 13.0, 50.0):
        layer.current_spacing = torch.full((2, 3), s)
        assert torch.allclose(layer(x), base(x))


def test_spacing_lora_needs_spacing():
    from models.spacing_lora import SpacingConditionedLoRALinear
    layer = SpacingConditionedLoRALinear(nn.Linear(4, 4))
    with pytest.raises(RuntimeError):
        layer(torch.randn(1, 4))


def test_film_modulates_lora_branch():
    # y = Wx + b + (alpha/r) * (gamma * B(A(x)) + beta), gamma = 1 + film_gamma
    from models.spacing_lora import SpacingConditionedLoRALinear, encode_spacing
    torch.manual_seed(0)
    base = nn.Linear(6, 8)
    layer = SpacingConditionedLoRALinear(base, r=2, alpha=1.0, spacing_encoding="log")
    for p in list(layer.lora_B.parameters()) + list(layer.spacing_film[-1].parameters()):
        nn.init.normal_(p)
    x = torch.randn(3, 4, 6)
    sp = torch.full((3, 3), 13.0)
    layer.current_spacing = sp
    film = layer.spacing_film(encode_spacing(sp, "log"))
    gamma, beta = 1 + film[:, :8], film[:, 8:]
    lora = layer.lora_B(layer.lora_A(x))
    expected = base(x) + 0.5 * (gamma[:, None] * lora + beta[:, None])
    assert torch.allclose(layer(x), expected, atol=1e-6)


def test_film_sweep_matches_the_module():
    # analyze_film_sweep.py reads gamma/beta from the state dict; it must give
    # what the layer itself computes
    from analyze_film_sweep import film, film_layers
    from models.spacing_lora import encode_spacing, inject_spacing_lora

    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Module() for _ in range(2)])
            for b in self.blocks:
                b.attn = nn.Module()
                b.attn.qkv, b.attn.proj = nn.Linear(8, 24), nn.Linear(8, 8)

    torch.manual_seed(0)
    enc = Enc()
    layers = inject_spacing_lora(enc, spacing_encoding="log")
    for m in enc.modules():
        if hasattr(m, "spacing_film"):
            nn.init.normal_(m.spacing_film[-1].weight)
            nn.init.normal_(m.spacing_film[-1].bias)
    sd = {"image_encoder." + k: v for k, v in enc.state_dict().items()}
    prefixes = film_layers(sd)
    assert len(prefixes) == 4
    assert [p.split("image_encoder.")[-1] for p in prefixes] == [
        "blocks.0.attn.qkv", "blocks.0.attn.proj", "blocks.1.attn.qkv", "blocks.1.attn.proj"]
    for s in (5.2, 50.0):
        g, b = film(sd, prefixes[0], s, "log")
        out = enc.blocks[0].attn.qkv.spacing_film(encode_spacing(torch.full((1, 3), s), "log"))[0]
        assert torch.allclose(g, 1 + out[:24], atol=1e-6)
        assert torch.allclose(b, out[24:], atol=1e-6)
    assert len(layers) == 4


# ---------------------------------------------------------------- loss
def test_loss_prefers_the_right_answer():
    from losses.combined_loss import HiPVesselLoss
    torch.manual_seed(0)
    target = torch.from_numpy(tube((24, 24, 24), r=3, center=(12, 12))).float()[None, None]
    sp = torch.tensor([[50.0, 50.0, 50.0]])
    loss = HiPVesselLoss()
    good_logits = (target * 2 - 1) * 6
    bad_logits = -good_logits
    l_good = loss(torch.sigmoid(good_logits), target, sp, logits=good_logits)
    l_bad = loss(torch.sigmoid(bad_logits), target, sp, logits=bad_logits)
    assert l_good < l_bad
    logits = torch.zeros_like(target, requires_grad=True)
    loss(torch.sigmoid(logits), target, sp, logits=logits).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_gradient_penalty_scales_with_spacing():
    from losses.combined_loss import HiPVesselLoss
    pred = torch.rand(1, 1, 8, 8, 8)
    target = torch.zeros_like(pred)
    loss = HiPVesselLoss(alpha=0.0, beta=1.0, use_cldice=False)
    base = loss(pred, target, torch.tensor([[1e9, 1e9, 1e9]]))  # penalty ~ 0
    l50 = loss(pred, target, torch.tensor([[50.0] * 3])) - base
    l5 = loss(pred, target, torch.tensor([[5.0] * 3])) - base
    assert l5 / l50 == pytest.approx(100.0, rel=1e-3)  # penalty ~ 1/spacing^2


# ---------------------------------------------------------------- post-processing
def test_binarize_threshold_and_min_size():
    from benchmark.postprocess import binarize
    prob = np.zeros((20, 20, 20), dtype=np.uint8)
    prob[2:12, 2:12, 2:12] = 200   # 1000 voxels
    prob[15:17, 15:17, 15:17] = 200  # 8 voxels
    prob[5, 15, 15] = 100
    assert binarize(prob, 0.5, 0).sum() == 1008
    assert binarize(prob, 0.3, 0).sum() == 1009
    assert binarize(prob, 0.5, 20).sum() == 1000
    assert binarize(prob, 0.5, 8).sum() == 1000  # a component of exactly min_size is removed
    assert binarize(prob, 0.5, 7).sum() == 1008


# ---------------------------------------------------------------- inference
class SignModel(nn.Module):
    """Logit = normalised intensity, so the prediction is 'brighter than the window mean'."""

    def forward(self, x, spacing):
        return [x]


class ConstModel(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x, spacing):
        return [torch.full_like(x, self.value)]


@pytest.mark.parametrize("resample_um", [None, 50.0])
@pytest.mark.parametrize("tta", [False, True])
def test_predict_roi_shapes_and_constants(resample_um, tta):
    from benchmark.infer import predict_roi
    img = np.random.default_rng(0).random((40, 36, 30)).astype(np.float32)
    for value, expected in ((5.0, 1), (-5.0, 0)):
        pred = predict_roi(ConstModel(value), img, 13.0, 16, "cpu", resample_um=resample_um, tta=tta)
        assert pred.shape == img.shape and pred.dtype == np.uint8
        assert np.all(pred == expected)
    prob = predict_roi(ConstModel(0.0), img, 13.0, 16, "cpu", resample_um=resample_um, tta=tta, return_prob=True)
    assert np.allclose(prob, 0.5, atol=1e-4)


def test_predict_roi_finds_a_bright_tube():
    from benchmark.infer import predict_roi
    g = tube((48, 48, 48), r=4, center=(24, 24))
    # background far below every window's mean, so 'brighter than the mean' is exactly the tube
    img = (2 * g - 1).astype(np.float32) * 5 + 0.05 * np.random.default_rng(0).standard_normal(g.shape).astype(np.float32)
    pred = predict_roi(SignModel(), img, 50.0, 32, "cpu")
    assert score_roi(pred, g, graph_stats=False)["dice"] > 0.95
    pred_tta = predict_roi(SignModel(), img, 50.0, 32, "cpu", tta=True)
    assert score_roi(pred_tta, g, graph_stats=False)["dice"] > 0.95


def test_spacing_reaches_the_model():
    # native: the model sees the true spacing; tell50: 50 um; fov50: the grid spacing (50 um)
    from benchmark.infer import predict_roi
    seen = []

    class Spy(nn.Module):
        def forward(self, x, spacing):
            seen.append(float(spacing[0, 0]))
            return [torch.zeros_like(x)]

    img = np.zeros((20, 20, 20), dtype=np.float32)
    for kw, want in (({}, 13.0), ({"spacing_um": 50.0}, 50.0), ({"resample_um": 50.0}, 50.0)):
        seen.clear()
        predict_roi(Spy(), img, 13.0, 16, "cpu", **kw)
        assert seen and all(s == pytest.approx(want) for s in seen), kw


def test_binarize_uses_26_connectivity():
    from benchmark.postprocess import binarize
    prob = np.zeros((10, 10, 10), dtype=np.uint8)
    prob[1, 1, 1] = prob[2, 2, 2] = 255  # touch only at a corner: one 26-connected component of 2
    assert binarize(prob, 0.5, 1).sum() == 2
    assert binarize(prob, 0.5, 2).sum() == 0


def test_saved_score_files():
    # every results/scores/*.json: right number of ROIs per test set, metrics in [0, 1]
    import glob
    import json
    n_rois = {"kidney_5": 8, "kidney_6": 8, "kidney_additional_voi_13um": 6, "kidney_1_voi": 6}
    files = glob.glob("results/scores/*.json")
    assert files
    for f in files:
        sets = json.load(open(f))["sets"]
        for sid, v in sets.items():
            if sid in n_rois:
                assert v["n_rois"] == n_rois[sid] == len(v["per_roi"]), (f, sid)
            for row in v["per_roi"]:
                for m in ("dice", "cldice", "nsd_t0", "nsd_t1"):
                    assert 0.0 <= row[m] <= 1.0, (f, sid, row["roi"], m)


# ---------------------------------------------------------------- training loop
def test_training_loop_runs_saves_and_resumes(tmp_path, monkeypatch):
    import json
    import train_lora_v2 as T

    class FakeData(torch.utils.data.IterableDataset):
        def __init__(self, *a, **k):
            super().__init__()

        def __iter__(self):
            g = torch.Generator().manual_seed(0)
            while True:
                m = (torch.rand(1, 16, 16, 16, generator=g) > 0.9).float()
                yield {"image": m + 0.1 * torch.randn(1, 16, 16, 16, generator=g), "mask": m,
                       "spacing": torch.tensor([50.0, 50.0, 50.0])}

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(1, 1, 3, padding=1)

        def forward(self, x, spacing):
            logits = self.conv(x)
            return torch.sigmoid(logits), logits

    monkeypatch.setattr(T, "SlabPatchDataset", FakeData)
    monkeypatch.setattr(T, "build", lambda args: Tiny())
    out = str(tmp_path / "run")
    args = T.get_parser().parse_args(["--model", "plain", "--out", out, "--epochs", "2", "--iters", "5",
                                      "--workers", "0", "--patch", "16", "--warmup", "1"])
    T.train(args)
    log = json.load(open(f"{out}/train_log.json"))
    assert [e["epoch"] for e in log] == [1, 2]
    assert all(np.isfinite(e["loss"]) and 0 < e["fg_fraction"] < 1 for e in log)
    st = torch.load(f"{out}/state.pth", map_location="cpu")
    assert st["epoch"] == 2 and all(v.device.type == "cpu" for v in st["model"].values())
    # resume to 3 epochs: only the third epoch runs
    args.epochs = 3
    T.train(args)
    assert [e["epoch"] for e in json.load(open(f"{out}/train_log.json"))] == [1, 2, 3]


def test_score_many_finds_runs(tmp_path):
    from benchmark.score_many import find_run
    (tmp_path / "a" / "preds" / "r1").mkdir(parents=True)
    (tmp_path / "b" / "r2").mkdir(parents=True)
    (tmp_path / "b" / "r2" / "run.json").write_text("{}")
    (tmp_path / "c" / "r3").mkdir(parents=True)
    assert find_run(str(tmp_path), "r1").endswith("preds/r1")
    assert find_run(str(tmp_path), "r2").endswith("b/r2")
    assert find_run(str(tmp_path), "r3") is None
