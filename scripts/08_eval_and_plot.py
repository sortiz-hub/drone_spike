"""Step 8: Evaluate trained policy and generate plots.

Requires a trained model from step 7 at models/ppo_intercept_final.zip
"""

from pathlib import Path

from drone_intercept.training.eval_policy import evaluate

model_path = "models/ppo_intercept_final.zip"
if not Path(model_path).exists():
    print(f"SKIP — no model found at {model_path}. Run step 7 first.")
    raise SystemExit(0)

results = evaluate(
    model_path=model_path,
    n_episodes=20,
    log_dir="logs/eval",
    plot=True,
    seed=0,
)

print(f"\nSuccess rate: {results['success_rate']:.0%}")
if results["success_rate"] > 0:
    print("PASS")
else:
    print("WARNING: 0% capture rate — may need more training steps")
