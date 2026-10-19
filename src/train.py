"""Fine-tune MMS-TTS-aka on the prepared Twi clips (run on a GPU, e.g. Colab).

Training itself is done by Y. Lacombe's finetune-hf-vits (MIT licence). This
script sets it up and makes one change to it:

  * freezing: the modules named under "freeze" in configs/train.json (by
    default the text encoder and duration predictor, which hold most of
    what the base model knows about Twi pronunciation and timing) are taken
    out of the optimizer, so only the acoustic side adapts to the new voice

Afterwards, `--export` turns every saved checkpoint into a normal VitsModel
folder so each one can be scored and the best one kept.

Usage:
  python src/train.py                  # set up, train, export checkpoints
  python src/train.py --export         # only export checkpoints of a finished run
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "runs/finetune-hf-vits"
HARNESS_URL = "https://github.com/ylacombe/finetune-hf-vits"
HARNESS_COMMIT = "6f3f51f4d667f5c3eef89484d151ffd39d2c2b89"  # Feb 2024, latest at time of writing

# The one line in run_vits_finetuning.py that hands parameters to the generator optimizer.
OPTIMIZER_LINE = """    gen_optimizer = torch.optim.AdamW(
        model.parameters(),"""
FROZEN_OPTIMIZER = """    for name in os.environ.get("KASA_FREEZE", "").split(","):
        if name:
            getattr(model, name).requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"kasa: frozen {os.environ.get('KASA_FREEZE')}; "
                f"training {sum(p.numel() for p in trainable):,} of {sum(p.numel() for p in model.parameters()):,} parameters")
    gen_optimizer = torch.optim.AdamW(
        trainable,"""


def run(cmd, cwd=None, env=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, check=True)


def set_up_harness():
    if not HARNESS.exists():
        run(["git", "clone", HARNESS_URL, HARNESS])
        run(["git", "checkout", "-q", HARNESS_COMMIT], cwd=HARNESS)
        # build the Cython alignment search, following the harness README
        (HARNESS / "monotonic_align/monotonic_align").mkdir(exist_ok=True)
        run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=HARNESS / "monotonic_align")

    script = HARNESS / "run_vits_finetuning.py"
    code = script.read_text()
    if "KASA_FREEZE" not in code:
        assert code.count(OPTIMIZER_LINE) == 1, "finetune-hf-vits changed; freezing patch no longer applies"
        script.write_text(code.replace(OPTIMIZER_LINE, FROZEN_OPTIMIZER))


def base_model_with_discriminator(path):
    """The released MMS-aka has no discriminator; GAN fine-tuning needs Meta's original one."""
    if not (path / "config.json").exists():
        run([sys.executable, "convert_original_discriminator_checkpoint.py",
             "--language_code", "aka", "--pytorch_dump_folder_path", path], cwd=HARNESS)


def export_checkpoints(out_dir):
    """checkpoint-N (accelerate state) -> exports/step-N (loadable with VitsModel)."""
    import torch
    from safetensors.torch import load_file
    from transformers import VitsModel

    for ckpt in sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])):
        target = out_dir / "exports" / f"step-{ckpt.name.split('-')[1]}"
        if target.exists():
            continue
        model = VitsModel.from_pretrained(out_dir)  # final model: right config and tokenizer
        # weights in the checkpoint are stored with weight norm still applied
        model.decoder.apply_weight_norm()
        for flow in model.flow.flows:
            torch.nn.utils.weight_norm(flow.conv_pre)
            torch.nn.utils.weight_norm(flow.conv_post)
        state = load_file(ckpt / "model.safetensors")
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not missing, f"{ckpt.name}: missing weights {missing[:5]}"
        model.decoder.remove_weight_norm()
        for flow in model.flow.flows:
            torch.nn.utils.remove_weight_norm(flow.conv_pre)
            torch.nn.utils.remove_weight_norm(flow.conv_post)
        model.save_pretrained(target)
        for f in ("tokenizer_config.json", "vocab.json", "special_tokens_map.json", "added_tokens.json"):
            if (out_dir / f).exists():
                (target / f).write_bytes((out_dir / f).read_bytes())
        print(f"exported {ckpt.name} -> {target.relative_to(ROOT)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=ROOT / "configs/train.json", type=Path)
    p.add_argument("--export", action="store_true", help="only export checkpoints of a finished run")
    args = p.parse_args()

    cfg = json.loads(args.config.read_text())
    freeze = cfg.pop("freeze", [])
    for key in ("model_name_or_path", "output_dir", "dataset_name"):
        cfg[key] = str(ROOT / cfg[key])
    out_dir = Path(cfg["output_dir"])

    if not args.export:
        set_up_harness()
        base_model_with_discriminator(Path(cfg["model_name_or_path"]))
        out_dir.mkdir(parents=True, exist_ok=True)
        harness_cfg = out_dir / "harness_config.json"
        harness_cfg.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        env = {**os.environ, "KASA_FREEZE": ",".join(freeze)}
        run(["accelerate", "launch", "run_vits_finetuning.py", harness_cfg], cwd=HARNESS, env=env)

    sys.path.insert(0, str(HARNESS))
    export_checkpoints(out_dir)


if __name__ == "__main__":
    main()
