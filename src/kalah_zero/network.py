from __future__ import annotations

from dataclasses import dataclass

from kalah_zero.encoding import encode_state, input_size
from kalah_zero.game import GameState


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


@dataclass(slots=True)
class NeuralEvaluator:
    model: KalahNet
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


def save_checkpoint(path: str, model: KalahNet, optimizer=None, step: int = 0) -> None:
    torch = _torch()
    payload = {
        "model_state": model.state_dict(),
        "pits": model.pits,
        "step": step,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str, device: str = "cpu") -> tuple[KalahNet, dict]:
    torch = _torch()
    payload = torch.load(path, map_location=device, weights_only=False)
    model = KalahNet(pits=payload.get("pits", 6)).to(device)
    model.load_state_dict(payload["model_state"])
    return model, payload
