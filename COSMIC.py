from __future__ import annotations

from pathlib import Path

from cosmic.config import COSMICConfig
from cosmic.pipeline import COSMICPipeline


def build_config(args) -> COSMICConfig:
    return COSMICConfig(
        dataname=args.dataname,
        run_dir=Path("ckpt") / args.dataname / "cosmic",
        mask_ratio=args.mask_ratio,
        hidden_dim=args.cosmic_hidden_dim,
        depth=args.cosmic_depth,
        num_heads=args.cosmic_num_heads,
        dropout=args.cosmic_dropout,
        epochs=args.cosmic_epochs,
        batch_size=args.cosmic_batch_size,
        learning_rate=args.cosmic_lr,
        weight_decay=args.cosmic_weight_decay,
        shap_estimator=args.shap_estimator,
        order_strategy=args.order_strategy,
        mice_rounds=args.mice_rounds,
        context_dim=args.cosmic_context_dim,
        num_trees=args.mice_trees,
        shap_num_samples=args.shap_num_samples,
        seed=args.seed,
    )


def main(args):
    config = build_config(args)
    pipeline = COSMICPipeline(config)
    print(pipeline.describe())

    if args.mode == "train":
        result = pipeline.train(device=args.device)
        print(f"[COSMIC][train] ordered columns: {result['ordered_columns']}")
        return

    pipeline.sample(
        device=args.device,
        num_samples=args.num_samples,
        save_path=args.save_path,
    )
