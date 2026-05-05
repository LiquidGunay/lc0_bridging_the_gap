"""Helpers for building LC0 optimal-vs-subpar rollout records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

try:
    import chess
    import chess.engine
except ImportError:  # pragma: no cover
    chess = None


@dataclass
class RolloutLine:
    """One principal variation returned by LC0 search."""

    move: str
    score_cp: int | None
    depth: int | None
    nodes: int | None
    pv: list[str]
    fens: list[str]
    multipv_rank: int | None = None
    seldepth: int | None = None
    nps: int | None = None
    hashfull: int | None = None
    tbhits: int | None = None
    wdl: dict[str, int] | None = None
    score_delta_cp: int | None = None
    raw_info_keys: list[str] | None = None
    activation_keys: list[str] | None = None

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class RolloutPairRecord:
    """Root position with the best LC0 line and selected subpar alternatives."""

    root_fen: str
    node_budget: int | None
    best: RolloutLine
    subpar: list[RolloutLine]
    root_history_fens: list[str] | None = None
    root_game_id: str | None = None
    root_game_index: int | None = None
    root_ply: int | None = None
    root_source: str | None = None
    root_record_id: str | None = None
    root_history_reconstructed: bool | None = None
    search_metadata: dict[str, Any] | None = None

    def to_json(self) -> dict:
        payload = {
            "root_fen": self.root_fen,
            "node_budget": self.node_budget,
            "best": self.best.to_json(),
            "subpar": [line.to_json() for line in self.subpar],
        }
        optional = {
            "root_history_fens": self.root_history_fens,
            "root_game_id": self.root_game_id,
            "root_game_index": self.root_game_index,
            "root_ply": self.root_ply,
            "root_source": self.root_source,
            "root_record_id": self.root_record_id,
            "root_history_reconstructed": self.root_history_reconstructed,
            "search": self.search_metadata,
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        return payload


def pv_to_fens(
    root: "chess.Board | str",
    pv: list["chess.Move"],
    *,
    max_depth: int | None = None,
) -> list[str]:
    """Replay a PV from the root and return root-plus-continuation FENs."""
    if chess is None:
        raise ImportError("python-chess is required for MCTS rollout extraction.")
    board = chess.Board(root) if isinstance(root, str) else root.copy(stack=False)
    fens = [board.fen()]
    moves = pv if max_depth is None else pv[:max_depth]
    for move in moves:
        if move not in board.legal_moves:
            break
        board.push(move)
        fens.append(board.fen())
    return fens


def board_from_root_history(
    fen: str,
    root_history_fens: list[str] | None = None,
) -> tuple["chess.Board", bool]:
    """Return a root board, reconstructing a short move stack from history when possible."""
    if chess is None:
        raise ImportError("python-chess is required for MCTS rollout extraction.")

    fallback = chess.Board(fen)
    history = [str(item) for item in (root_history_fens or []) if str(item)]
    if not history:
        return fallback, False
    if history[-1] != fen:
        history.append(fen)
    if len(history) == 1:
        return fallback, False

    try:
        board = chess.Board(history[0])
        for target_fen in history[1:]:
            matched_move = None
            for move in board.legal_moves:
                candidate = board.copy(stack=True)
                candidate.push(move)
                if candidate.fen() == target_fen:
                    matched_move = move
                    break
            if matched_move is None:
                return fallback, False
            board.push(matched_move)
    except ValueError:
        return fallback, False
    if board.fen() != fen:
        return fallback, False
    return board, True


def activation_records_for_line(
    line: RolloutLine,
    *,
    line_id: str,
    history_len: int = 8,
    root_history_fens: list[str] | None = None,
    root_game_id: str | None = None,
    root_game_index: int | None = None,
    root_ply: int | None = None,
    root_source: str | None = None,
    root_record_id: str | None = None,
) -> list[dict]:
    """Return activation records for a rollout line and attach stable keys."""
    if chess is None:
        raise ImportError("python-chess is required for MCTS rollout extraction.")
    if history_len < 1:
        raise ValueError("history_len must be >= 1")
    if not line.fens:
        return []

    root_fen = line.fens[0]
    base_history = [str(fen) for fen in (root_history_fens or []) if str(fen)]
    if not base_history or base_history[-1] != root_fen:
        base_history.append(root_fen)

    keys = []
    records = []
    resolved_root_ply = chess.Board(root_fen).ply() if root_ply is None else int(root_ply)
    for idx, fen in enumerate(line.fens):
        key = f"{line_id}:{idx:03d}"
        continuation = line.fens[1 : idx + 1]
        keys.append(key)
        record = {
            "fen": fen,
            "history_fens": (base_history + continuation)[-history_len:],
            "game_id": line_id,
            "ply": resolved_root_ply + idx,
            "activation_key": key,
            "trajectory_index": idx,
        }
        if root_game_id is not None:
            record["root_game_id"] = root_game_id
        if root_game_index is not None:
            record["root_game_index"] = int(root_game_index)
        if root_ply is not None:
            record["root_ply"] = resolved_root_ply
        if root_source is not None:
            record["root_source"] = root_source
        if root_record_id is not None:
            record["root_record_id"] = root_record_id
        records.append(record)
    line.activation_keys = keys
    return records


def score_cp_from_info(info: dict, *, turn: "chess.Color") -> int | None:
    """Return centipawn score from the side-to-move perspective."""
    score = info.get("score")
    if score is None:
        return None
    pov = score.pov(turn)
    return pov.score(mate_score=100000)


def wdl_from_info(info: dict, *, turn: "chess.Color") -> dict[str, int] | None:
    """Return WDL counts from the side-to-move perspective when the engine exposes them."""
    wdl = info.get("wdl")
    if wdl is None:
        return None
    try:
        pov = wdl.pov(turn)
    except AttributeError:
        pov = wdl
    return {
        "wins": int(pov.wins),
        "draws": int(pov.draws),
        "losses": int(pov.losses),
    }


def line_from_info(
    root: "chess.Board",
    info: dict,
    *,
    max_depth: int | None = None,
) -> RolloutLine | None:
    """Convert a python-chess analysis info dict into a serializable line."""
    pv = list(info.get("pv") or [])
    if not pv:
        return None
    return RolloutLine(
        move=pv[0].uci(),
        score_cp=score_cp_from_info(info, turn=root.turn),
        depth=info.get("depth"),
        nodes=info.get("nodes"),
        pv=[move.uci() for move in (pv if max_depth is None else pv[:max_depth])],
        fens=pv_to_fens(root, pv, max_depth=max_depth),
        multipv_rank=info.get("multipv"),
        seldepth=info.get("seldepth"),
        nps=info.get("nps"),
        hashfull=info.get("hashfull"),
        tbhits=info.get("tbhits"),
        wdl=wdl_from_info(info, turn=root.turn),
        raw_info_keys=sorted(str(key) for key in info.keys()),
    )


def build_rollout_pair_record(
    engine: "chess.engine.SimpleEngine",
    fen: str,
    limit: "chess.engine.Limit",
    *,
    multipv: int = 4,
    min_delta_cp: int = 25,
    max_delta_cp: int | None = None,
    max_depth: int | None = None,
    node_budget: int | None = None,
    root_history_fens: list[str] | None = None,
    root_game_id: str | None = None,
    root_game_index: int | None = None,
    root_ply: int | None = None,
    root_source: str | None = None,
    root_record_id: str | None = None,
    search_metadata: dict[str, Any] | None = None,
) -> RolloutPairRecord | None:
    """Run MultiPV search and select meaningful subpar alternatives."""
    if chess is None:
        raise ImportError("python-chess is required for MCTS rollout extraction.")
    board, history_reconstructed = board_from_root_history(fen, root_history_fens)
    infos = engine.analyse(board, limit, multipv=multipv)
    if isinstance(infos, dict):
        infos = [infos]
    infos = sorted(infos, key=lambda item: item.get("multipv", 1))
    best = line_from_info(board, infos[0], max_depth=max_depth) if infos else None
    if best is None:
        return None

    subpar = []
    for info in infos[1:]:
        line = line_from_info(board, info, max_depth=max_depth)
        if line is None:
            continue
        if best.score_cp is not None and line.score_cp is not None:
            delta = best.score_cp - line.score_cp
            line.score_delta_cp = delta
            if delta < min_delta_cp:
                continue
            if max_delta_cp is not None and delta > max_delta_cp:
                continue
        subpar.append(line)
    if not subpar:
        return None
    return RolloutPairRecord(
        root_fen=fen,
        node_budget=node_budget,
        best=best,
        subpar=subpar,
        root_history_fens=root_history_fens,
        root_game_id=root_game_id,
        root_game_index=root_game_index,
        root_ply=root_ply,
        root_source=root_source,
        root_record_id=root_record_id,
        root_history_reconstructed=history_reconstructed,
        search_metadata=search_metadata,
    )
