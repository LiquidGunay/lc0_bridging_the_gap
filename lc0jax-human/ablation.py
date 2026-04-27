import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from lc0jaxhuman.training.jepa import (
    JEPAConfig,
    create_jepa_components,
    build_synthetic_transition_batch,
    train_step
)
from lc0jaxhuman.weights import load_pb_gz, map_bt4_weights
from flax import nnx

def run_ablation():
    print("Loading weights...")
    weights_path = "models/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz"
    bundle = load_pb_gz(weights_path)
    bt4_params = map_bt4_weights(bundle)

    # We will test sizes: 1, 2, 4 layers
    sizes = [1, 2, 4]

    # We will test variants: Baseline, QK_Gain, XSA, Muon, All
    variants = [
        {"name": "Baseline", "kwargs": {}},
        {"name": "QK_Gain", "kwargs": {"use_qk_gain": True}},
        {"name": "Muon", "kwargs": {"use_muon": True}},
        {"name": "All", "kwargs": {"use_qk_gain": True, "use_muon": True}},
    ]

    print("Generating batch...")
    # Generate a fixed synthetic batch
    batch_size = 32
    raw_batch = build_synthetic_transition_batch(batch_size)
    from lc0jaxhuman.training.jepa import build_transition_batch
    train_batch = build_transition_batch(raw_batch)

    train_batch = jax.tree.map(lambda x: jnp.asarray(x), train_batch)

    results = []

    for size in sizes:
        for variant in variants:
            print(f"Training size {size} variant {variant['name']}")
            config = JEPAConfig(
                token_dim=128,
                mlp_dim=512,
                num_layers=size,
                num_heads=4,
                **variant['kwargs']
            )
            model, optimizer = create_jepa_components(bt4_params, config, seed=42)

            # compiled train step
            @nnx.jit
            def step(model, optimizer, batch):
                return train_step(model, optimizer, batch)

            loss_hist = []
            for i in range(10):
                loss, aux = step(model, optimizer, train_batch)
                loss_hist.append(float(loss))

            final_loss = loss_hist[-1]
            param_count = sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
            results.append({
                "Layers": size,
                "Params": param_count,
                "Variant": variant["name"],
                "Final_Loss": final_loss
            })

    df = pd.DataFrame(results)
    print(df.to_string())
    df.to_csv("ablation_results.csv", index=False)

    # Generate plotting script
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x='Params', y='Final_Loss', hue='Variant', marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Scaling Laws: JEPA Loss vs Params')
    plt.savefig('scaling_plot.png')
    print("Saved plot to scaling_plot.png")

if __name__ == "__main__":
    run_ablation()