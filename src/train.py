"""Fine-tune MMS-TTS-aka on the prepared Twi clips (run on a GPU, e.g. Colab).

Training itself is done by Y. Lacombe's finetune-hf-vits (MIT licence). This
script sets it up and makes two changes to it:

  * freezing: the modules named under "freeze" in configs/train.json (by
    default the text encoder and duration predictor, which hold most of
    what the base model knows about Twi pronunciation and timing) are taken
    out of the optimizer, so only the acoustic side adapts to the new voice
  * plotting: the harness draws its training charts with matplotlib's
    tostring_rgb(), which matplotlib 3.10 removed; swapped for buffer_rgba()

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
import sysconfig
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

OLD_PLOT = "data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)"
NEW_PLOT = "data = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()"


def run(cmd, cwd=None, env=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    # Colab points MPLBACKEND at its notebook plugin, which our separate
    # environment doesn't have; the harness only needs plain image output
    env = {**(env or os.environ), "MPLBACKEND": "Agg"}
    subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, check=True)


def set_up_harness():
    if not HARNESS.exists():
        run(["git", "clone", HARNESS_URL, HARNESS])
        run(["git", "checkout", "-q", HARNESS_COMMIT], cwd=HARNESS)
    # build the Cython alignment search for *this* Python, following the harness README
    built = HARNESS / "monotonic_align/monotonic_align"
    if not list(built.glob("core" + sysconfig.get_config_var("EXT_SUFFIX"))):
        built.mkdir(exist_ok=True)
        run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=HARNESS / "monotonic_align")

    patch(HARNESS / "run_vits_finetuning.py", OPTIMIZER_LINE, FROZEN_OPTIMIZER, count=1)
    patch(HARNESS / "utils/plot.py", OLD_PLOT, NEW_PLOT, count=2)


def patch(path, old, new, count):
    code = path.read_text()
    if new in code:
        return
    assert code.count(old) == count, f"{path.name} changed upstream; patch no longer applies"
    path.write_text(code.replace(old, new))


def base_model_with_discriminator(path):
    """The released MMS-aka has no discriminator; GAN fine-tuning needs Meta's original one."""
    if not (path / "config.json").exists():
        run([sys.executable, "convert_original_discriminator_checkpoint.py",
             "--language_code", "aka", "--pytorch_dump_folder_path", path], cwd=HARNESS)


# the two ways PyTorch stores a weight-normed weight w = g * v / |v|
WEIGHT_NORM_LAYOUTS = [(".weight_g", ".weight_v"),
                       (".parametrizations.weight.original0", ".parametrizations.weight.original1")]


def match_layout(ckpt, target):
    """Rewrite checkpoint weights into the key layout the target model uses.

    The training harness and transformers' VitsModel don't agree on how weight
    norm is stored: as plain `weight`, as (weight_g, weight_v), or as
    parametrizations (original0, original1). Each target key is filled from
    whichever form the checkpoint has, converting between them as needed.
    """
    import torch

    def plain(prefix):
        if prefix + ".weight" in ckpt:
            return ckpt[prefix + ".weight"]
        for g_key, v_key in WEIGHT_NORM_LAYOUTS:
            if prefix + g_key in ckpt:
                g, v = ckpt[prefix + g_key], ckpt[prefix + v_key]
                return torch._weight_norm(v, g, next((i for i, n in enumerate(g.shape) if n != 1), 0))
        raise KeyError(prefix)

    out = {}
    for key, like in target.items():
        if key in ckpt:
            out[key] = ckpt[key]
            continue
        for g_key, v_key in WEIGHT_NORM_LAYOUTS:
            if key.endswith(g_key):  # target wants g: |w| over every dim except the one g keeps
                w = plain(key[: -len(g_key)])
                dim = next((i for i, n in enumerate(like.shape) if n != 1), 0)
                out[key] = torch.norm_except_dim(w, 2, dim)
                break
            if key.endswith(v_key):  # target wants v: w itself is a valid direction
                out[key] = plain(key[: -len(v_key)])
                break
        else:
            out[key] = plain(key[: -len(".weight")])
    return out


def export_checkpoints(out_dir):
    """checkpoint-N (accelerate state) -> exports/step-N (loadable with VitsModel)."""
    from safetensors.torch import load_file
    from transformers import VitsModel

    for ckpt in sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])):
        target = out_dir / "exports" / f"step-{ckpt.name.split('-')[1]}"
        if target.exists():
            continue
        model = VitsModel.from_pretrained(out_dir)  # final model: right config and tokenizer
        ckpt_state = load_file(ckpt / "model.safetensors")
        model.load_state_dict(match_layout(ckpt_state, model.state_dict()))
        model.save_pretrained(target)
        for f in ("tokenizer_config.json", "vocab.json", "special_tokens_map.json", "added_tokens.json"):
            if (out_dir / f).exists():
                (target / f).write_bytes((out_dir / f).read_bytes())
        print(f"exported {ckpt.name} -> {target.relative_to(ROOT) if target.is_relative_to(ROOT) else target}")


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
        import torch
        # without CUDA, keep accelerate off Apple's MPS backend: it lacks the FFT used for spectrograms
        device = [] if torch.cuda.is_available() else ["--cpu"]
        run([sys.executable, "-m", "accelerate.commands.launch", "--num_processes", "1", *device,
             "--mixed_precision", "fp16" if cfg.get("fp16") else "no",
             "run_vits_finetuning.py", harness_cfg], cwd=HARNESS, env=env)

    sys.path.insert(0, str(HARNESS))
    export_checkpoints(out_dir)


if __name__ == "__main__":
    main()
