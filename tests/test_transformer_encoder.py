"""Tests for the transformer encoder."""

from itertools import pairwise
from pathlib import Path

import pytest
import torch
from torch import Tensor

from envs.emitter import BatchedEmitterEnv, EnvConfig
from models.transformer_encoder import EncoderConfig, TransformerEncoder, load_config

ROOT = Path(__file__).resolve().parents[1]


def make_model(
    coordinate_dim: int = 2,
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    num_layers: int = 2,
    max_steps: int = 7,
) -> TransformerEncoder:
    """Build the small encoder used by these tests."""
    config = EncoderConfig(8, num_layers, 2, 16)
    return TransformerEncoder(
        config=config,
        coordinate_dim=coordinate_dim,
        max_steps=max_steps,
    ).to(device=device, dtype=dtype)


def observation() -> dict[str, Tensor]:
    """Return two concrete point sets with heterogeneous receiver demands."""
    return {
        "points": torch.tensor(
            [[[0.0, 0.1], [0.5, 0.6], [1.0, 1.1]], [[0.2, 0.3], [0.7, 0.8], [1.2, 1.3]]]
        ),
        "weights": torch.tensor([[1.0, 0.5, 1.5], [0.75, 1.25, 0.25]]),
        "selected": torch.tensor([[True, False, True], [False, True, False]]),
        "received_signal": torch.tensor([[0.8, 0.9, 1.0], [0.4, 0.75, 1.0]]),
        "demands": torch.tensor([[0.6, 1.0, 1.2], [0.5, 0.8, 1.1]]),
        "steps_remaining": torch.tensor([4, 2], dtype=torch.int64),
        "contributions": torch.tensor(
            [
                [[1.0, 0.1, 0.2], [0.1, 1.0, 0.3], [0.2, 0.3, 1.0]],
                [[1.0, 0.4, 0.5], [0.4, 1.0, 0.6], [0.5, 0.6, 1.0]],
            ]
        ),
    }




def test_features() -> None:
    """Point features keep coordinates, scalars, and the signed demand margin in order."""
    model = make_model()
    obs = {
        "points": torch.tensor([[[1.25, -2.0], [3.0, 4.0]]]),
        "weights": torch.tensor([[0.25, 0.75]]),
        "selected": torch.tensor([[True, False]]),
        "received_signal": torch.tensor([[0.8, 1.3]]),
        "demands": torch.tensor([[1.2, 1.0]]),
        "steps_remaining": torch.tensor([4]),
    }
    seen: list[Tensor] = []
    hook = model.input_projection[0].register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone())
    )
    model(obs)
    hook.remove()

    expected = torch.tensor(
        [[[1.25, -2.0, 0.25, 1.0, 0.8, -0.4], [3.0, 4.0, 0.75, 0.0, 1.3, 0.3]]]
    )
    torch.testing.assert_close(seen[0], expected)


def test_global_features_include_mean_and_fraction() -> None:
    """Global features contain the point mean and remaining-step fraction."""
    model = make_model(coordinate_dim=2, dtype=torch.float64)
    obs = {
        key: value.to(dtype=torch.float64) if value.is_floating_point() else value
        for key, value in observation().items()
    }
    seen: list[Tensor] = []
    hook = model.global_projection[0].register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone())
    )
    point_embeddings, _ = model(obs)
    hook.remove()

    expected = torch.cat(
        (
            point_embeddings.mean(dim=1),
            obs["steps_remaining"].to(torch.float64).unsqueeze(-1) / 7,
        ),
        dim=-1,
    )
    torch.testing.assert_close(seen[0], expected)


@pytest.mark.parametrize("coordinate_dim", [1, 2, 3])
def test_shapes_and_reuse(coordinate_dim: int) -> None:
    """One model handles different point shapes, including a single batch row or point."""
    model = make_model(coordinate_dim)
    for batch_size, num_points in [(2, 4), (1, 3), (3, 1), (1, 1), (3, 5), (2, 2)]:
        obs = {
            "points": torch.zeros(batch_size, num_points, coordinate_dim),
            "weights": torch.ones(batch_size, num_points),
            "selected": torch.zeros(batch_size, num_points, dtype=torch.bool),
            "received_signal": torch.ones(batch_size, num_points),
            "demands": torch.full((batch_size, num_points), 0.5),
            "steps_remaining": torch.zeros(batch_size, dtype=torch.int64),
        }
        points, global_embedding = model(obs)
        assert points.shape == (batch_size, num_points, 8)
        assert global_embedding.shape == (batch_size, 8)


def test_reordering_points() -> None:
    """Reordering points reorders point outputs but keeps the global output fixed."""
    model = make_model(dtype=torch.float64).eval()
    obs = {
        key: value.to(dtype=torch.float64) if value.is_floating_point() else value
        for key, value in observation().items()
    }
    order = torch.tensor([2, 0, 1])
    reordered = dict(obs)
    for key in ("points", "weights", "selected", "received_signal", "demands"):
        reordered[key] = obs[key][:, order]
    reordered["contributions"] = obs["contributions"][:, order][:, :, order]

    with torch.no_grad():
        points, global_embedding = model(obs)
        reordered_points, reordered_global = model(reordered)
    torch.testing.assert_close(reordered_points, points[:, order], rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(reordered_global, global_embedding, rtol=1e-5, atol=1e-8)


def test_batch_rows_are_independent_and_inputs_are_untouched() -> None:
    """Changing one row leaves the other row's outputs unchanged; forward does not alter inputs."""
    model = make_model(dtype=torch.float64).eval()
    obs = {
        key: value.to(dtype=torch.float64) if value.is_floating_point() else value
        for key, value in observation().items()
    }
    before = {key: value.clone() for key, value in obs.items()}
    changed = {key: value.clone() for key, value in obs.items()}
    changed["points"][1] += 10
    changed["weights"][1] *= 2
    changed["selected"][1] = ~changed["selected"][1]
    changed["received_signal"][1] += 0.4
    changed["demands"][1] += 0.2
    changed["steps_remaining"][1] = 6

    with torch.no_grad():
        points, global_embedding = model(obs)
        changed_points, changed_global = model(changed)
    torch.testing.assert_close(changed_points[0], points[0])
    torch.testing.assert_close(changed_global[0], global_embedding[0])
    for key, value in obs.items():
        assert torch.equal(value, before[key])


def test_layers_are_independent_and_seed_reproducible() -> None:
    """The same seed reproduces models while separate layers own separate weights."""
    config = EncoderConfig(8, 3, 2, 16)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(314159)
        first = TransformerEncoder(config=config, coordinate_dim=2, max_steps=7)
        torch.manual_seed(314159)
        second = TransformerEncoder(config=config, coordinate_dim=2, max_steps=7)

        assert all(
            torch.equal(left, right)
            for left, right in zip(
                first.state_dict().values(), second.state_dict().values()
            )
        )
        for left_layer, right_layer in pairwise(first.layers):
            for left, right in zip(left_layer.parameters(), right_layer.parameters()):
                assert left.data_ptr() != right.data_ptr()
        assert any(
            not torch.equal(left, right)
            for left, right in zip(
                first.layers[0].parameters(), first.layers[1].parameters()
            )
            if left.ndim >= 2
        )


@pytest.mark.parametrize("loss_kind", ["h", "g"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_gradients(loss_kind: str, dtype: torch.dtype, device_name: str) -> None:
    """Point and global losses backpropagate on each supported dtype and device."""
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_name)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260906)
        model = make_model(dtype=dtype, device=device).train()
    obs = {
        key: value.to(device=device, dtype=dtype)
        if value.is_floating_point()
        else value.to(device=device)
        for key, value in observation().items()
    }
    points, global_embedding = model(obs)
    output = points if loss_kind == "h" else global_embedding
    weights = torch.linspace(0.25, 1.75, output.numel(), dtype=dtype, device=device)
    (output * weights.reshape_as(output)).sum().backward()

    modules = [model.input_projection, *model.layers, model.output_norm]
    if loss_kind == "g":
        modules.append(model.global_projection)
    for module in modules:
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(gradient.abs().sum() > 0 for gradient in gradients)


def test_environment_reset_observation() -> None:
    """A real environment reset observation can pass through the encoder."""
    points = torch.tensor(
        [[[0.0], [0.5], [1.0], [1.5]], [[0.2], [0.7], [1.2], [1.7]]],
        dtype=torch.float32,
    )
    config = EnvConfig(
        num_emitters=2,
        max_exchanges=1,
        max_steps=4,
        seed=17,
        device="cpu",
    )
    demands = torch.tensor(
        [[0.5, 0.7, 1.1, 1.4], [0.6, 0.8, 1.0, 1.2]], dtype=torch.float32
    )
    env = BatchedEmitterEnv(
        points,
        torch.ones(2, 4),
        config,
        demands=demands,
    )
    obs, info = env.reset(seed=23)
    contributions = obs["contributions"]
    assert contributions.shape == (2, 4, 4)
    torch.testing.assert_close(
        contributions.diagonal(dim1=-2, dim2=-1), torch.ones(2, 4), rtol=0, atol=0
    )
    torch.testing.assert_close(contributions[:, 0, 1], torch.full((2,), -0.5).exp())
    encoded_points, global_embedding = make_model(1, max_steps=4)(obs)
    assert encoded_points.shape == (2, 4, 8)
    assert global_embedding.shape == (2, 8)
    assert info["objective"].shape == (2,)
