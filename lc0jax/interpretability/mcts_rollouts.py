"""Helpers for building LC0 optimal-vs-subpar rollout records."""

from __future__ import annotations

from dataclasses import asdict, dataclass

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

    def to_json(self) -> dict:
        return {
            "root_fen": self.root_fen,
            "node_budget": self.node_budget,
            "best": self.best.to_json(),
            "subpar": [line.to_json() for line in self.subpar],
        }


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


def activation_records_for_line(
    line: RolloutLine,
    *,
    line_id: str,
    history_len: int = 8,
) -> list[dict]:
    """Return activation records for a rollout line and attach stable keys."""
    if chess is None:
        raise ImportError("python-chess is required for MCTS rollout extraction.")
    if history_len < 1:
        raise ValueError("history_len must be >= 1")
    keys = []
    records = []
    root_ply = chess.Board(line.fens[0]).ply() if line.fens else 0
    for idx, fen in enumerate(line.fens):
        key = f"{line_id}:{idx:03d}"
        keys.append(key)
        records.append(
            {
                "fen": fen,
                "history_fens": line.fens[max(0, idx - history_len + 1) : idx + 1],
                "game_id": line_id,
                "ply": root_ply + idx,
                "activation_key": key,
                "trajectory_index": idx,
            }
        )
    line.activation_keys = keys
    return records


def score_cp_from_info(info: dict, *, turn: "chess.Color") -> int | None:
    """Return centipawn score from the side-to-move perspective."""
    score = info.get("score")
    if score is None:
        return None
    pov = score.pov(turn)
    return pov.score(mate_score=100000)


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
) -> RolloutPairRecord | None:
    """Run MultiPV search and select meaningful subpar alternatives."""
    if chess is None:
        raise ImportError("python-chess is required for MCTS rollout extraction.")
    board = chess.Board(fen)
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
    )
