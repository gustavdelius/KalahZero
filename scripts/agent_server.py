#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import _path  # noqa: F401
from kalah_zero.batched_mcts import BatchedMCTS
from kalah_zero.game import GameState
from kalah_zero.mcts import MCTS
from kalah_zero.network import NeuralEvaluator, load_checkpoint


def build_state(board: list[int], current_player: int, pits: int, use_fast_game: bool):
    state_cls = GameState
    if use_fast_game:
        from kalah_zero.fast_game import FastGameState

        state_cls = FastGameState
    return state_cls(tuple(board), current_player=current_player, pits=pits)


def build_search(simulations: int, use_batched_mcts: bool, eval_batch_size: int):
    if use_batched_mcts:
        return BatchedMCTS(simulations=simulations, batch_size=eval_batch_size)
    return MCTS(simulations=simulations)


class AgentServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_cls, args: argparse.Namespace) -> None:
        super().__init__(server_address, handler_cls)
        model, _ = load_checkpoint(args.checkpoint)
        self.evaluator = NeuralEvaluator(model)
        self.pits = args.pits
        self.simulations = args.simulations
        self.use_batched_mcts = args.batched_mcts
        self.eval_batch_size = args.eval_batch_size
        self.use_fast_game = args.fast_game


class AgentHandler(BaseHTTPRequestHandler):
    server: AgentServer

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(404)
            return
        self._send_json({"ok": True})

    def do_POST(self) -> None:
        if self.path != "/move":
            self.send_error(404)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length) or b"{}")
            state = build_state(
                board=list(payload["board"]),
                current_player=int(payload["current_player"]),
                pits=int(payload.get("pits", self.server.pits)),
                use_fast_game=self.server.use_fast_game,
            )
            if state.is_terminal():
                self._send_json({"error": "terminal position"}, status=400)
                return
            search = build_search(
                simulations=int(payload.get("simulations", self.server.simulations)),
                use_batched_mcts=self.server.use_batched_mcts,
                eval_batch_size=self.server.eval_batch_size,
            )
            result = search.search(state, self.server.evaluator)
            action = result.select_action(temperature=0.0)
            self._send_json(
                {
                    "action": action,
                    "policy": result.policy,
                    "visits": result.visits,
                    "value": result.value,
                    "simulations": search.simulations,
                }
            )
        except Exception as error:  # noqa: BLE001 - server should return JSON errors to the browser.
            self._send_json({"error": str(error)}, status=400)

    def log_message(self, format: str, *args) -> None:
        print(f"{self.client_address[0]} - {format % args}")

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve trained KalahZero moves to the browser board.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--pits", type=int, default=6)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--batched-mcts", action="store_true")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--fast-game", action="store_true")
    args = parser.parse_args()

    server = AgentServer((args.host, args.port), AgentHandler, args)
    print(
        f"serving {args.checkpoint} at http://{args.host}:{args.port}/move "
        f"(simulations={args.simulations})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping agent server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
