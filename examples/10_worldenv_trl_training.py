"""WorldEnv + TRL GRPO Training — OpenEnv Hackathon Example.

Demonstrates training an LLM agent on WorldEnv environments using
HF TRL's GRPOTrainer with a custom rollout function.

WorldEnv is deployed at: https://huggingface.co/spaces/jacob-valdez/worldenv

The same world model underlies 6 scenarios:
  corporate_employee, corporate_executive, macro_policy,
  logistics_robot, disaster_robot, climate_monitor

Install:
    pip install openenv-core trl transformers torch

Run:
    python examples/10_worldenv_trl_training.py
"""

import json
import random

# ── Connect to WorldEnv on HF Spaces ────────────────────────────
from openenv.core import EnvClient

WORLDENV_URL = "https://jacob-valdez-worldenv.hf.space"

# Import client (install from repo or local)
try:
    from worldenv import WorldEnv, WorldAction
except ImportError:
    # Fallback: minimal inline client for demo
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from worldenv import WorldEnv, WorldAction


def demo_env_interaction():
    """Quick demo — connect to WorldEnv and run a few steps."""
    print("Connecting to WorldEnv at:", WORLDENV_URL)

    with WorldEnv(base_url=WORLDENV_URL) as env:
        result = env.reset()
        obs = result.observation

        print(f"\nScenario: {obs.scenario_name}")
        print(f"Info: {obs.info}")
        print(f"Observable fields ({len(obs.obs_field_names)}):")
        for field, val in list(obs.visible_fields.items())[:5]:
            print(f"  {field}: {val:.4f}")

        # Step 1: Intervene on an action field
        if obs.act_field_names:
            act_field = obs.act_field_names[0]
            action = WorldAction(
                action_type="intervene",
                target_field=act_field,
                value=0.8,
            )
            result = env.step(action)
            print(f"\nAfter intervening on {act_field}:")
            print(f"  Reward: {result.reward:.4f}")
            print(f"  Step: {result.observation.step_count}")

        # Step 2: Predict a field
        if obs.obs_field_names:
            pred_field = obs.obs_field_names[0]
            action = WorldAction(
                action_type="predict",
                target_field=pred_field,
                value=0.5,
            )
            result = env.step(action)
            print(f"\nAfter predicting {pred_field}:")
            print(f"  Reward: {result.reward:.4f}")


# ── TRL GRPO Integration ─────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI agent operating in a world simulation.
You observe economic, physical, and social fields from the world model.
Your goal is to take actions that maximize reward.

Available action types:
- observe: Look at a field (no reward, just info)
- predict: Predict a field value (reward ∝ accuracy)
- intervene: Set an action field (drives dynamics)
- step: Advance time

Respond with ONLY a JSON action, no markdown:
{"action_type": "intervene", "target_field": "firm.strategy.rd_intensity", "value": 0.8}
"""


def format_observation(obs) -> str:
    """Format a WorldObservation as a text prompt for the LLM."""
    fields_str = "\n".join(
        f"  {k}: {v:.4f}" for k, v in list(obs.visible_fields.items())[:8]
    )
    return (
        f"Scenario: {obs.scenario_name}\n"
        f"Step: {obs.step_count}\n"
        f"Observable fields:\n{fields_str}\n"
        f"Actionable fields: {obs.act_field_names}\n"
        f"Status: {obs.info}"
    )


def parse_action(text: str, obs) -> "WorldAction":
    """Parse LLM output into a WorldAction."""
    text = text.strip()
    # Strip markdown fences if present
    if "```" in text:
        text = text.split("```")[1].strip()
        if text.startswith("json"):
            text = text[4:].strip()

    try:
        d = json.loads(text)
        return WorldAction(
            action_type=d.get("action_type", "step"),
            target_field=d.get("target_field", ""),
            value=float(d.get("value", 0.0)),
        )
    except Exception:
        # Default to a random intervention on a valid field
        if obs.act_field_names:
            return WorldAction(
                action_type="intervene",
                target_field=random.choice(obs.act_field_names),
                value=random.uniform(-1, 1),
            )
        return WorldAction(action_type="step")


def run_grpo_training():
    """Full TRL GRPO training loop with WorldEnv rollouts."""
    try:
        import torch
        from trl import GRPOConfig, GRPOTrainer
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("TRL not installed. Run: pip install trl transformers torch")
        return

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
    )

    env_client = WorldEnv(base_url=WORLDENV_URL)
    env_client.connect()

    def rollout_func(prompts, trainer):
        """Custom rollout: run WorldEnv episodes, collect rewards."""
        all_prompt_ids = []
        all_completion_ids = []
        all_rewards = []

        for _ in prompts:
            # Reset env (random scenario)
            result = env_client.reset()
            obs = result.observation

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"World environment observation:\n"
                        f"{format_observation(obs)}\n\n"
                        "Choose your action:"
                    ),
                },
            ]

            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(
                trainer.model.device
            )

            with torch.no_grad():
                outputs = trainer.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.8,
                )

            completion_ids = outputs[0][inputs.input_ids.shape[1]:]
            completion_text = tokenizer.decode(
                completion_ids, skip_special_tokens=True,
            )

            # Parse and execute action
            action = parse_action(completion_text, obs)
            episode_reward = 0.0

            for step in range(10):
                step_result = env_client.step(action)
                episode_reward += step_result.reward or 0.0
                if step_result.observation.done:
                    break
                # Subsequent steps: random intervene
                if obs.act_field_names:
                    action = WorldAction(
                        action_type="intervene",
                        target_field=random.choice(obs.act_field_names),
                        value=random.uniform(-1, 1),
                    )
                else:
                    action = WorldAction(action_type="step")

            all_prompt_ids.append(inputs.input_ids[0])
            all_completion_ids.append(completion_ids)
            all_rewards.append(episode_reward)

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "rewards": all_rewards,
        }

    training_args = GRPOConfig(
        output_dir="outputs/worldenv-agent",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        logging_steps=1,
        save_steps=100,
        max_completion_length=128,
        num_generations=4,
        bf16=True,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        rollout_func=rollout_func,
    )

    print("Starting GRPO training on WorldEnv...")
    trainer.train()
    env_client.disconnect()

    print("Training complete!")
    print("WorldEnv Space: https://huggingface.co/spaces/jacob-valdez/worldenv")


if __name__ == "__main__":
    import sys

    if "--train" in sys.argv:
        run_grpo_training()
    else:
        print("WorldEnv Demo")
        print("=============")
        print()
        print("HF Space:  https://huggingface.co/spaces/jacob-valdez/worldenv")
        print("Docs:      https://jacobfv.github.io/general-unified-world-modeling/environments/")
        print("GitHub:    https://github.com/JacobFV/general-unified-world-modeling")
        print()
        print("Running env interaction demo...")
        demo_env_interaction()
        print()
        print("To run TRL GRPO training: python examples/10_worldenv_trl_training.py --train")
