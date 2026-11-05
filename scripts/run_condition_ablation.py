"""条件模式消融：固定 beta 与数据划分，比较 CVAE01-04 与最终模型。

用法：

```bash
python scripts/run_condition_ablation.py --data data/real/npy --out runs/condition_ablation
```

四种 CVAE 分别对应论文中的不同条件注入方式；`M_Rm_CVAE02` 作为
最终模型参考。结果写入 `condition_ablation.csv`。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _pipeline import METRIC_KEYS, add_common_args, print_table, run_experiment, write_summary_csv

DEFAULT_MODELS: tuple[str, ...] = ("CVAE01", "CVAE02", "CVAE03", "CVAE04", "M_Rm_CVAE02")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare CVAE condition modes")
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
        write_summary_csv(out_root / "condition_ablation.csv", rows, fieldnames)

    print("\n===== condition ablation summary =====")
    print_table(rows, fieldnames)


if __name__ == "__main__":
    main()
