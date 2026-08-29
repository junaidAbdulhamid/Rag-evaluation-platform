"""Phase 2 CLI: load the golden dataset and print a summary.

    python -m scripts.show_dataset
    python -m scripts.show_dataset --slice multi_document
    python -m scripts.show_dataset --category refunds --full

Its real job is to prove the requirement "the system can iterate through the entire
evaluation dataset" - everything here is driven by `for example in dataset`.
"""

from __future__ import annotations

import argparse
from collections import Counter

from app.evaluation.dataset import EvalDataset, load_eval_dataset

RULE = "=" * 78


def counts(dataset: EvalDataset, key) -> Counter:
    tally: Counter = Counter()
    for example in dataset:
        value = key(example)
        if isinstance(value, list):
            tally.update(value or ["(none)"])
        else:
            tally.update([value or "(none)"])
    return tally


def print_distribution(title: str, tally: Counter) -> None:
    print(f"\n{title}")
    for name, n in sorted(tally.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {name:<16} {n}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the golden eval dataset.")
    parser.add_argument("--path", default=None, help="Override the dataset path.")
    parser.add_argument("--slice", dest="slice_label", default=None, help="Only show this slice.")
    parser.add_argument("--category", default=None, help="Only show this category.")
    parser.add_argument("--full", action="store_true", help="Print every question in full.")
    args = parser.parse_args()

    dataset = load_eval_dataset(args.path)
    if args.slice_label:
        dataset = dataset.filter_by_slice(args.slice_label)
    if args.category:
        dataset = dataset.filter_by_category(args.category)

    print(f"{RULE}\nGOLDEN EVAL DATASET  ({len(dataset)} examples)\n{RULE}")

    print_distribution("By category:", counts(dataset, lambda e: e.category))
    print_distribution("By difficulty:", counts(dataset, lambda e: e.difficulty))
    print_distribution("By slice label:", counts(dataset, lambda e: e.slices))

    print(f"\n{RULE}\nEXAMPLES\n{RULE}")
    for example in dataset:
        question = example.question if args.full else example.question[:70]
        print(f"\n{example.id}  [{example.category}/{example.difficulty}]  slices={example.slices}")
        print(f"  Q: {question}")
        if args.full:
            print(f"  A: {example.expected_answer}")
            print(f"  relevant_document_ids: {example.relevant_document_ids}")


if __name__ == "__main__":
    main()
