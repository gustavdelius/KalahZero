from __future__ import annotations

from dataclasses import dataclass

from kalah_zero.encoding import encode_state, input_size
from kalah_zero.game import GameState


NetworkModel = object


def _torch():
    import torch

    return torch


def _nn():
    import torch.nn as nn

    return nn


class KalahNet(_nn().Module):
    def __init__(self, pits: int = 6, hidden_size: int = 128) -> None:
        super().__init__()
        nn = _nn()
        self.pits = pits
        self.hidden_size = hidden_size
        self.model_type = "mlp"
        self.residual_blocks = 0
        self.trunk = nn.Sequential(
            nn.Linear(input_size(pits), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, pits)
        self.value_head = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())

    def forward(self, x):
        features = self.trunk(x)
        return self.policy_head(features), self.value_head(features).squeeze(-1)


class ResidualBlock(_nn().Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        nn = _nn()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.activation = nn.ReLU()

    def forward(self, h):
        return self.activation(h + self.layers(h))


class ResidualKalahNet(_nn().Module):
    def __init__(self, pits: int = 6, hidden_size: int = 128, residual_blocks: int = 3) -> None:
        super().__init__()
        nn = _nn()
        self.pits = pits
        self.hidden_size = hidden_size
        self.model_type = "residual"
        self.residual_blocks = residual_blocks
        self.input_layer = nn.Sequential(
            nn.Linear(input_size(pits), hidden_size),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_size) for _ in range(residual_blocks)]
        )
        self.policy_head = nn.Linear(hidden_size, pits)
        self.value_head = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())

    def forward(self, x):
        features = self.blocks(self.input_layer(x))
        return self.policy_head(features), self.value_head(features).squeeze(-1)


def create_model(
    model_type: str = "mlp",
    pits: int = 6,
    hidden_size: int = 128,
    residual_blocks: int = 3,
):
    if model_type == "mlp":
        return KalahNet(pits=pits, hidden_size=hidden_size)
    if model_type == "residual":
        return ResidualKalahNet(
            pits=pits,
            hidden_size=hidden_size,
            residual_blocks=residual_blocks,
        )
    raise ValueError(f"unknown model type {model_type!r}")


@dataclass(slots=True)
class NeuralEvaluator:
    model: object
    device: str = "cpu"

    def evaluate(self, state: GameState) -> tuple[list[float], float]:
        return self.evaluate_batch([state])[0]

    def evaluate_batch(self, states: list[GameState]) -> list[tuple[list[float], float]]:
        torch = _torch()
        if not states:
            return []
        if self.model.training:
            self.model.eval()
        with torch.inference_mode():
            x = torch.stack([encode_state(state) for state in states]).to(self.device)
            logits, values = self.model(x)
            logits = logits.detach().cpu()
            values = values.detach().cpu()

        results: list[tuple[list[float], float]] = []
        for state, row, value in zip(states, logits, values):
            mask = torch.full_like(row, float("-inf"))
            legal = state.legal_actions()
            if legal:
                for action in legal:
                    mask[action] = 0.0
                probs = torch.softmax(row + mask, dim=0)
            else:
                probs = torch.zeros_like(row)
            results.append((probs.tolist(), float(value.item())))
        return results


def save_checkpoint(path: str, model: NetworkModel, optimizer=None, step: int = 0) -> None:
    torch = _torch()
    payload = {
        "model_state": model.state_dict(),
        "pits": model.pits,
        "model_type": getattr(model, "model_type", "mlp"),
        "hidden_size": getattr(model, "hidden_size", 128),
        "residual_blocks": getattr(model, "residual_blocks", 0),
        "step": step,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str, device: str = "cpu") -> tuple[NetworkModel, dict]:
    torch = _torch()
    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload.get("config", {})
    model_type = payload.get("model_type", config.get("model_type", "mlp"))
    hidden_size = payload.get("hidden_size", config.get("hidden_size", 128))
    residual_blocks = payload.get(
        "residual_blocks",
        config.get("residual_blocks", 3 if model_type == "residual" else 0),
    )
    model = create_model(
        model_type=model_type,
        pits=payload.get("pits", config.get("pits", 6)),
        hidden_size=hidden_size,
        residual_blocks=residual_blocks,
    ).to(device)
    model.load_state_dict(payload["model_state"])
    return model, payload
