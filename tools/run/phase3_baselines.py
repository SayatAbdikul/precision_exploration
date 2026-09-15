"""Verify classifier pairs and rescore original detector predictions on screen 1k."""
from tools.phase3.baselines import baseline
from tools.phase3.common import ROOT, campaign, digest, write


def main():
    plan = campaign()
    report = {"schema_version": "phase3-baselines-1.0.0", "campaign_sha256": digest(plan), "models": {}}
    for model in plan["models"]:
        report["models"][model] = baseline(model, plan)
        write(ROOT / "results/summaries/phase3-baselines.json", report)
        print(model, "frozen paired baseline verified", flush=True)


if __name__ == "__main__":
    main()
