"""Beta 扫描：固定最终模型结构，遍历 KL 权重 beta。

用法：

```bash
python scripts/run_beta_sweep.py --data data/real/npy --out runs/beta_sweep
```

每个 beta 训练一个独立检查点，随后用 RBFI 在评估集上计算误差，
结果汇总到 `beta_sweep.csv`。只保留论文使用的显式参数，不自动调参。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _pipeline import METRIC_KEYS, add_common_args, print_table, run_experiment, write_summary_csv

DEFAULT_BETAS: tuple[float, ...] = (1000.0, 100.0, 10.0, 1.0, 0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6, 1e-7)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep beta for the final Beta-CVAE")
    add_common_args(parser)
    parser.add_argument("--model", default="M_Rm_CVAE02", help="固定使用的模型结构")
    parser.add_argument("--betas", type=float, nargs="+", default=list(DEFAULT_BETAS))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_root = Path(args.out)
    rows = []
    fieldnames = ["beta", *METRIC_KEYS]

    for beta in args.betas:
        print(f"\n===== beta={beta:g} =====")
        summary = run_experiment(
            args.data,
            out_root / f"beta_{beta:g}",
            model=args.model,
            beta=beta,
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
        rows.append({"beta": beta, **{key: summary[key] for key in METRIC_KEYS}})
        # 每轮都落盘，便于中断后查看已有结果
        write_summary_csv(out_root / "beta_sweep.csv", rows, fieldnames)

    print("\n===== beta sweep summary =====")
    print_table(rows, fieldnames)


if __name__ == "__main__":
    main()
