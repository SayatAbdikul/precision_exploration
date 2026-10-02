"""Experiment B2 reconstruction baseline (lane L6): AdaRound fits, evaluations and the report.

  fit     continue one resumable fit (exit code 0 = complete, 3 = time budget reached, run again)
  eval    evaluate the missing arms of one group for one model on the 1k screen
  plan    list the layers of a model with the memory a fit holds for each (no GPU)
  report  write the summaries under results/summaries/b2-recon-v1/ (no GPU)
"""
import argparse
import json
import os

# Must be fixed before the first CUDA context for deterministic cuBLAS operations.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

MODELS = ("resnet18", "mobilenet_v2", "mobilenet_v3_large")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--wformat", required=True)
    p.add_argument("--rule", default="mse_per_channel")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--input", choices=("fp32_activations", "b2_activations"), default="fp32_activations")
    p.add_argument("--aformat")
    p.add_argument("--recipe", default="default_no_bias_correction")
    p.add_argument("--budget", type=float, default=600.0)
    p.add_argument("--max-layers", type=int)
    p.add_argument("--cap-gib", type=float, default=3.0)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p = sub.add_parser("eval")
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--group", required=True)
    p.add_argument("--only", nargs="*")
    p.add_argument("--budget", type=float, default=600.0)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p = sub.add_parser("plan")
    p.add_argument("--model", choices=MODELS, required=True)
    p = sub.add_parser("report")
    p.add_argument("--tag", required=True)
    p.add_argument("--out")
    p.add_argument("--figure", action="store_true")
    args = parser.parse_args()
    if args.command == "fit":
        from tools.experiment_b2_recon.fit import run_fit
        b2 = args.input == "b2_activations"
        return run_fit(args.model, args.wformat, args.rule, args.seed, args.device, budget_seconds=args.budget,
                       reconstruction_input=args.input, aformat=(args.aformat or args.wformat) if b2 else None,
                       recipe_name=args.recipe if b2 else None, cap_gib=args.cap_gib, max_layers=args.max_layers)
    if args.command == "eval":
        from tools.experiment_b2_recon.evaluate import run_group
        return run_group(args.model, args.group, args.device, budget_seconds=args.budget, only=args.only)
    if args.command == "plan":
        import torch
        from tools.experiment_b.classifier import configure, load_model
        from tools.experiment_b2_recon.engine import consumer_activation, weight_layers
        configure("cpu")
        graph, _, _ = load_model(args.model, "cpu")
        shapes = {}

        class Shapes(torch.fx.Interpreter):
            def run_node(self, node):
                result = super().run_node(node)
                shapes[node.name] = (tuple(self.fetch_args_kwargs_from_env(node)[0][0].shape[1:])
                                     if node.op == "call_module" else None, tuple(result.shape[1:]))
                return result

        with torch.inference_mode():
            Shapes(graph).run(torch.zeros(1, 3, 224, 224))
        total = 0
        for index, (node, module) in enumerate(weight_layers(graph)):
            source, out = shapes[node.name]
            gib_in = 1024 * 4 * float(torch.tensor(source).prod()) / 2 ** 30
            gib_out = 1024 * 4 * float(torch.tensor(out).prod()) / 2 ** 30
            total += module.weight.numel()
            print(json.dumps({"index": index, "node": node.name, "weight": list(module.weight.shape),
                              "act": consumer_activation(graph, node)[0], "in": list(source), "out": list(out),
                              "held_gib": round(gib_in + min(gib_in, gib_out), 2)}))
        print(json.dumps({"layers": index + 1, "weights": total}))
        return 0
    from tools.experiment_b2_recon.report import main as report
    return report(["--tag", args.tag] + (["--out", args.out] if args.out else []) + (["--figure"] if args.figure else []))


if __name__ == "__main__":
    raise SystemExit(main())
