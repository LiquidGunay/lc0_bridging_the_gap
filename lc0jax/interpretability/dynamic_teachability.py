"""Teachability curriculum artifacts for dynamic concepts."""

from __future__ import annotations

from typing import Any

import numpy as np


def _copy_row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    bulky = {"rank", "policy_logits", "teacher_logits", "logits"}
    return {key: value for key, value in row.items() if key not in bulky}


def _required_row_value(row: dict[str, Any], key: str, *, group: str) -> Any:
    value = row.get(key)
    if value in (None, ""):
        raise ValueError(f"{group} row missing required key '{key}'")
    return value


def _curriculum_record(
    row: dict[str, Any],
    *,
    group: str,
    split: str,
    direction_key: str,
    reverse: bool,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "group": group,
        "split": split,
        "rank": int(_required_row_value(row, "rank", group=group)),
        "pair_index": int(_required_row_value(row, "index", group=group)),
        "score": float(_required_row_value(row, "score", group=group)),
        "projection_score": float(row.get("projection_score", row["score"])),
        "direction_key": direction_key,
        "reverse": bool(reverse),
        "root_fen": _required_row_value(row, "root_fens", group=group),
        "target_move": _required_row_value(row, "best_moves", group=group),
        "contrast_move": _required_row_value(row, "subpar_moves", group=group),
        "provenance": provenance,
        "metadata": _copy_row_metadata(row),
    }
    return record


def teachability_curriculum_records(
    prototypes_report: dict[str, Any],
    *,
    max_prototypes: int | None = None,
    max_controls: int | None = None,
) -> list[dict[str, Any]]:
    """Build JSONL-ready curriculum records from a prototypes report."""
    if max_prototypes is not None and max_prototypes < 0:
        raise ValueError("max_prototypes must be >= 0")
    if max_controls is not None and max_controls < 0:
        raise ValueError("max_controls must be >= 0")

    split = str(prototypes_report.get("split", "train"))
    direction_key = str(prototypes_report.get("direction_key", "direction"))
    reverse = bool(prototypes_report.get("reverse", False))
    provenance = {
        key: prototypes_report[key]
        for key in ("pairs", "concept", "seed", "score_mode")
        if key in prototypes_report
    }
    prototypes = list(prototypes_report.get("prototypes", []))
    controls = list(prototypes_report.get("random_controls", []))
    if max_prototypes is not None:
        prototypes = prototypes[:max_prototypes]
    if max_controls is not None:
        controls = controls[:max_controls]

    records = [
        _curriculum_record(
            row,
            group="prototype",
            split=split,
            direction_key=direction_key,
            reverse=reverse,
            provenance=provenance,
        )
        for row in prototypes
    ]
    records.extend(
        _curriculum_record(
            row,
            group="random_control",
            split=split,
            direction_key=direction_key,
            reverse=reverse,
            provenance=provenance,
        )
        for row in controls
    )
    return records


def _softmax(logits: np.ndarray, *, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    shifted = scaled - scaled.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _mean_kl_from_teacher_probs(
    teacher_probs: np.ndarray,
    student_logits: np.ndarray,
    *,
    temperature: float,
    epsilon: float = 1e-12,
) -> float:
    student_probs = _softmax(student_logits, temperature=temperature)
    teacher = np.clip(np.asarray(teacher_probs, dtype=np.float64), epsilon, 1.0)
    student = np.clip(student_probs, epsilon, 1.0)
    return float(np.mean(np.sum(teacher * (np.log(teacher) - np.log(student)), axis=-1)))


def _top1_overlap(teacher_logits: np.ndarray, student_logits: np.ndarray) -> float:
    if teacher_logits.shape[0] == 0:
        return float("nan")
    teacher_top = np.argmax(teacher_logits, axis=-1)
    student_top = np.argmax(student_logits, axis=-1)
    return float(np.mean(teacher_top == student_top))


def _as_index_array(
    indices: np.ndarray | list[int] | tuple[int, ...] | None,
    *,
    row_count: int,
    key: str,
) -> np.ndarray:
    if indices is None:
        arr = np.arange(row_count, dtype=np.int64)
    else:
        arr = np.asarray(indices, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return arr
    if np.any(arr < 0) or np.any(arr >= row_count):
        raise IndexError(f"{key} contains row indices outside [0, {row_count})")
    return arr


def _validate_policy_arrays(
    features: np.ndarray,
    teacher_logits: np.ndarray,
) -> tuple[int, int, int]:
    x = np.asarray(features)
    y = np.asarray(teacher_logits)
    if x.ndim != 2:
        raise ValueError(f"features must be rank-2, got {x.shape}")
    if y.ndim != 2:
        raise ValueError(f"teacher_logits must be rank-2, got {y.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"feature/logit row mismatch: {x.shape[0]} vs {y.shape[0]}")
    if x.shape[0] == 0:
        raise ValueError("at least one row is required")
    return int(x.shape[0]), int(x.shape[1]), int(y.shape[1])


def curriculum_pair_indices(
    records: list[dict[str, Any]],
    *,
    group: str,
    max_records: int | None = None,
) -> np.ndarray:
    """Return row indices from curriculum records for one group."""
    if max_records is not None and max_records < 0:
        raise ValueError("max_records must be >= 0")
    rows = [int(record["pair_index"]) for record in records if record.get("group") == group]
    if max_records is not None:
        rows = rows[:max_records]
    return np.asarray(rows, dtype=np.int64)


def _adapter_logits(params: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    x = (np.asarray(features, dtype=np.float32) - params["feature_mean"]) / params["feature_std"]
    return (x @ params["w1"]) @ params["w2"] + params["bias"]


def train_low_rank_policy_adapter(
    features: np.ndarray,
    teacher_logits: np.ndarray,
    *,
    train_indices: np.ndarray | list[int] | tuple[int, ...] | None = None,
    hidden_dim: int = 32,
    steps: int = 200,
    batch_size: int = 32,
    learning_rate: float = 1e-2,
    l2: float = 1e-4,
    temperature: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Train a low-rank policy adapter on frozen features with KL distillation.

    The adapter is ``features -> hidden_dim -> policy_logits`` and is intended
    as a lightweight teachability proxy over stored activations or pair-level
    feature rows. Full-scale runs should use GPU-backed JAX, while unit tests
    keep the matrices tiny.
    """
    import jax
    import jax.numpy as jnp

    row_count, input_dim, output_dim = _validate_policy_arrays(features, teacher_logits)
    if hidden_dim < 1:
        raise ValueError("hidden_dim must be >= 1")
    if steps < 0:
        raise ValueError("steps must be >= 0")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    train_rows = _as_index_array(train_indices, row_count=row_count, key="train_indices")
    if train_rows.size == 0:
        raise ValueError("at least one train row is required")

    x_np = np.asarray(features, dtype=np.float32)
    y_np = _softmax(np.asarray(teacher_logits, dtype=np.float32), temperature=temperature).astype(
        np.float32
    )
    train_x_np = x_np[train_rows]
    feature_mean = train_x_np.mean(axis=0).astype(np.float32)
    feature_std = train_x_np.std(axis=0).astype(np.float32)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)
    x_np = (x_np - feature_mean) / feature_std

    key = jax.random.PRNGKey(seed)
    key_w1, key_w2 = jax.random.split(key)
    w1_scale = 1.0 / np.sqrt(max(input_dim, 1))
    w2_scale = 1.0 / np.sqrt(max(hidden_dim, 1))
    params = {
        "w1": jax.random.normal(key_w1, (input_dim, hidden_dim), dtype=jnp.float32) * w1_scale,
        "w2": jax.random.normal(key_w2, (hidden_dim, output_dim), dtype=jnp.float32) * w2_scale,
        "bias": jnp.zeros((output_dim,), dtype=jnp.float32),
    }
    x = jnp.asarray(x_np)
    y = jnp.asarray(y_np)

    def loss_fn(model_params, batch_x, batch_y):
        logits = (batch_x @ model_params["w1"]) @ model_params["w2"] + model_params["bias"]
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        cross_entropy = -jnp.mean(jnp.sum(batch_y * log_probs, axis=-1))
        penalty = l2 * (
            jnp.sum(model_params["w1"] ** 2) + jnp.sum(model_params["w2"] ** 2)
        )
        return cross_entropy + penalty

    @jax.jit
    def train_step(model_params, batch_x, batch_y):
        loss, grads = jax.value_and_grad(loss_fn)(model_params, batch_x, batch_y)
        updated = jax.tree_util.tree_map(
            lambda value, grad: value - learning_rate * grad,
            model_params,
            grads,
        )
        return updated, loss

    train_x = x[train_rows]
    train_y = y[train_rows]
    initial_loss = float(loss_fn(params, train_x, train_y))
    rng = np.random.default_rng(seed)
    last_loss = initial_loss
    for _ in range(steps):
        batch_rows = rng.choice(train_rows, size=batch_size, replace=train_rows.size < batch_size)
        params, loss = train_step(params, x[batch_rows], y[batch_rows])
        last_loss = float(loss)
    final_full_loss = float(loss_fn(params, train_x, train_y))

    params_np = {
        "w1": np.asarray(params["w1"]),
        "w2": np.asarray(params["w2"]),
        "bias": np.asarray(params["bias"]),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
    }
    return {
        "params": params_np,
        "history": {
            "initial_loss": initial_loss,
            "final_full_loss": final_full_loss,
            "last_minibatch_loss": last_loss,
            "steps": int(steps),
            "train_rows": int(train_rows.size),
            "hidden_dim": int(hidden_dim),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "l2": float(l2),
            "temperature": float(temperature),
        },
    }


def evaluate_policy_adapter(
    params: dict[str, np.ndarray],
    features: np.ndarray,
    teacher_logits: np.ndarray,
    *,
    indices: np.ndarray | list[int] | tuple[int, ...] | None = None,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Evaluate a trained adapter against teacher policy logits."""
    row_count, _, _ = _validate_policy_arrays(features, teacher_logits)
    rows = _as_index_array(indices, row_count=row_count, key="indices")
    if rows.size == 0:
        raise ValueError("at least one evaluation row is required")
    teacher = np.asarray(teacher_logits, dtype=np.float32)[rows]
    student = _adapter_logits(params, np.asarray(features, dtype=np.float32)[rows])
    teacher_probs = _softmax(teacher, temperature=temperature)
    return {
        "rows": int(rows.size),
        "kl_teacher_student": _mean_kl_from_teacher_probs(
            teacher_probs,
            student,
            temperature=temperature,
        ),
        "top1_overlap": _top1_overlap(teacher, student),
        "teacher_top1": [int(value) for value in np.argmax(teacher, axis=-1)[:20]],
        "student_top1": [int(value) for value in np.argmax(student, axis=-1)[:20]],
    }


def teachability_lift_report(
    features: np.ndarray,
    teacher_logits: np.ndarray,
    curriculum_records: list[dict[str, Any]],
    *,
    eval_features: np.ndarray | None = None,
    eval_teacher_logits: np.ndarray | None = None,
    eval_indices: np.ndarray | list[int] | tuple[int, ...] | None = None,
    max_prototypes: int | None = None,
    max_controls: int | None = None,
    hidden_dim: int = 32,
    steps: int = 200,
    batch_size: int = 32,
    learning_rate: float = 1e-2,
    l2: float = 1e-4,
    temperature: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Train prototype/control adapters and report teachability lift."""
    row_count, input_dim, output_dim = _validate_policy_arrays(features, teacher_logits)
    if eval_features is None:
        eval_features = features
    if eval_teacher_logits is None:
        eval_teacher_logits = teacher_logits
    eval_row_count, eval_input_dim, eval_output_dim = _validate_policy_arrays(
        eval_features,
        eval_teacher_logits,
    )
    if eval_input_dim != input_dim:
        raise ValueError(f"eval feature dimension mismatch: {eval_input_dim} vs {input_dim}")
    if eval_output_dim != output_dim:
        raise ValueError(f"eval policy-logit dimension mismatch: {eval_output_dim} vs {output_dim}")
    prototype_rows = curriculum_pair_indices(
        curriculum_records,
        group="prototype",
        max_records=max_prototypes,
    )
    control_rows = curriculum_pair_indices(
        curriculum_records,
        group="random_control",
        max_records=max_controls,
    )
    prototype_rows = _as_index_array(
        prototype_rows,
        row_count=row_count,
        key="prototype curriculum",
    )
    control_rows = _as_index_array(
        control_rows,
        row_count=row_count,
        key="random-control curriculum",
    )
    budget_match = int(prototype_rows.size) == int(control_rows.size)

    if eval_indices is None:
        if eval_features is features and eval_teacher_logits is teacher_logits:
            train_union = set(
                int(value) for value in np.concatenate([prototype_rows, control_rows])
            )
            eval_rows = np.asarray(
                [idx for idx in range(row_count) if idx not in train_union],
                dtype=np.int64,
            )
            if eval_rows.size == 0:
                eval_rows = np.arange(row_count, dtype=np.int64)
        else:
            eval_rows = np.arange(eval_row_count, dtype=np.int64)
    else:
        eval_rows = _as_index_array(eval_indices, row_count=eval_row_count, key="eval_indices")

    def fit_group(name: str, rows: np.ndarray, group_seed: int) -> dict[str, Any]:
        if rows.size == 0:
            return {"available": False, "reason": "no_curriculum_rows"}
        trained = train_low_rank_policy_adapter(
            features,
            teacher_logits,
            train_indices=rows,
            hidden_dim=hidden_dim,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            l2=l2,
            temperature=temperature,
            seed=group_seed,
        )
        params = trained["params"]
        return {
            "available": True,
            "group": name,
            "train_rows": int(rows.size),
            "train_indices_preview": [int(value) for value in rows[:20]],
            "history": trained["history"],
            "train": evaluate_policy_adapter(
                params,
                features,
                teacher_logits,
                indices=rows,
                temperature=temperature,
            ),
            "eval": evaluate_policy_adapter(
                params,
                eval_features,
                eval_teacher_logits,
                indices=eval_rows,
                temperature=temperature,
            ),
        }

    prototype = fit_group("prototype", prototype_rows, seed)
    random_control = fit_group("random_control", control_rows, seed)
    lift: dict[str, Any] = {
        "available": False,
        "budget_matched": budget_match,
        "prototype_train_rows": int(prototype_rows.size),
        "random_control_train_rows": int(control_rows.size),
    }
    if prototype.get("available") and random_control.get("available"):
        prototype_eval = prototype["eval"]
        control_eval = random_control["eval"]
        if budget_match:
            lift.update(
                {
                    "available": True,
                    "top1_overlap_lift": float(
                        prototype_eval["top1_overlap"] - control_eval["top1_overlap"]
                    ),
                    "kl_reduction": float(
                        control_eval["kl_teacher_student"]
                        - prototype_eval["kl_teacher_student"]
                    ),
                }
            )
        else:
            lift["reason"] = "train_row_budget_mismatch"

    return {
        "method": "low_rank_policy_adapter_kl",
        "num_rows": int(row_count),
        "input_dim": int(input_dim),
        "num_policy_logits": int(output_dim),
        "eval_num_rows": int(eval_row_count),
        "eval_rows": int(eval_rows.size),
        "eval_indices_preview": [int(value) for value in eval_rows[:20]],
        "hidden_dim": int(hidden_dim),
        "steps": int(steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "l2": float(l2),
        "temperature": float(temperature),
        "seed": int(seed),
        "prototype": prototype,
        "random_control": random_control,
        "lift": lift,
    }
