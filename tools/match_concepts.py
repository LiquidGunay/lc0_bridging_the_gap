"""Match unsupervised concept directions to Lichess puzzle tags."""

import argparse
import json
import numpy as np

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concepts", required=True, help="Path to discovered concept_direction.npz")
    parser.add_argument("--activations", required=True, help="Path to puzzle activations .npz or directory")
    parser.add_argument("--tags", required=True, help="Path to puzzle tags .jsonl file")
    parser.add_argument("--out", required=True, help="Output JSON report path")
    parser.add_argument("--top-tags", type=int, default=5, help="Number of top tags to report per concept")
    args = parser.parse_args()

    # Load concept directions
    concept_data = np.load(args.concepts)
    direction = concept_data["direction"]  # Shape: (d,) or (d, k)
    if direction.ndim == 1:
        direction = direction.reshape(-1, 1)

    # Load puzzle activations
    import pathlib
    p = pathlib.Path(args.activations)
    files = [p] if p.is_file() else sorted(p.glob("*.npz"))

    activations_list = []
    for file in files:
        data = np.load(file, allow_pickle=True)
        activations_list.append(data["embeddings"])

    if not activations_list:
        raise RuntimeError("No activations found")
    activations = np.concatenate(activations_list, axis=0)

    # Load tags
    puzzle_tags = []
    with open(args.tags, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                puzzle_tags.append(json.loads(line)["Themes"])

    if len(activations) != len(puzzle_tags):
        print(f"Warning: Number of activations ({len(activations)}) does not match number of tags ({len(puzzle_tags)}). Truncating to the minimum.")
        min_len = min(len(activations), len(puzzle_tags))
        activations = activations[:min_len]
        puzzle_tags = puzzle_tags[:min_len]

    # Project activations onto concept directions
    # activations: (N, d), direction: (d, k)
    # projections: (N, k)
    projections = activations @ direction

    # Aggregate scores per tag
    from collections import defaultdict

    k = direction.shape[1]
    report = {}

    for concept_idx in range(k):
        tag_scores = defaultdict(list)
        for i, tags in enumerate(puzzle_tags):
            score = float(projections[i, concept_idx])
            for tag in tags:
                tag_scores[tag].append(score)

        # Calculate average score for each tag
        tag_avg_scores = {}
        for tag, scores in tag_scores.items():
            if len(scores) >= 5: # Minimum occurrence filter
                tag_avg_scores[tag] = sum(scores) / len(scores)

        # Sort tags by average score
        sorted_tags = sorted(tag_avg_scores.items(), key=lambda item: item[1], reverse=True)
        top_tags = sorted_tags[:args.top_tags]
        bottom_tags = sorted_tags[-args.top_tags:][::-1] # Lowest scoring tags

        report[f"concept_{concept_idx}"] = {
            "top_tags": [{"tag": t, "avg_score": s, "count": len(tag_scores[t])} for t, s in top_tags],
            "bottom_tags": [{"tag": t, "avg_score": s, "count": len(tag_scores[t])} for t, s in bottom_tags]
        }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Match report written to {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
