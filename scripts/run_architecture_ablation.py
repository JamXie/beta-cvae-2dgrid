"""结构消融：比较多分支（M-VAE01-03）、残差（R-VAE01-03）与最终模型。

用法：

```bash
python scripts/run_architecture_ablation.py --data data/real/npy --out runs/architecture_ablation
```

模型名与论文表述的对应关系见 `docs/METHOD.md`：
M-VAE0x -> MultiBranchVAE0x，R-VAE0x -> ResidualVAE0x。
结果写入 `architecture_ablation.csv`。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _pipeline import METRIC_KEYS, add_common_args, print_table, run_experiment, write_summary_csv

DEFAULT_MODELS: tuple[str, ...] = (
    "MultiBranchVAE01",
    "MultiBranchVAE02",
    "MultiBranchVAE03",
    "ResidualVAE01",
    "ResidualVAE02",
    "ResidualVAE03",
    "M_Rm_CVAE02",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare multi-branch / residual / final models")
    add_common_args(parser)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--beta", type=float, default=1e-6)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_root = Path(args.out)
    rows = []
    fieldnames = ["model", "beta", *METRIC_KEYS]

    for model in args.models:
        print(f"\n===== model={model} =====")
        summary = run_experiment(
            args.data,
            out_root / model,
            model=model,
            beta=args.beta,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_batches=args.max_batches,
            fit_split=args.fit_split,
            eval_split=args.eval_split,
            epsilon=args.epsilon,
            device=args.device,
            seed=args.seed,
        )
        rows.append({"model": model, "beta": args.beta, **{key: summary[key] for key in METRIC_KEYS}})
        write_summary_csv(out_root / "architecture_ablation.csv", rows, fieldnames)

    print("\n===== architecture ablation summary =====")
    print_table(rows, fieldnames)


if __name__ == "__main__":
    main()
