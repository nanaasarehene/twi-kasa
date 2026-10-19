"""Step 0: confirm the environment can run the pipeline.

Usage: python src/check_setup.py
"""
import importlib
import platform
import sys

REQUIRED = ["torch", "torchaudio", "transformers", "datasets", "accelerate",
            "soundfile", "librosa", "jiwer", "yaml"]
# the training harness only works below these major versions
MAX_MAJOR = {"transformers": 4, "datasets": 3}


def main() -> int:
    print(f"Python   : {sys.version.split()[0]} ({platform.machine()})")
    missing = []
    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "")
            if name in MAX_MAJOR and int(version.split(".")[0]) > MAX_MAJOR[name]:
                print(f"  --  {name:<13} {version} is too new (need {MAX_MAJOR[name]}.x)")
                missing.append(name)
                continue
            print(f"  ok  {name:<13} {version}")
        except ImportError as exc:
            print(f"  --  {name:<13} MISSING ({exc})")
            missing.append(name)

    if "torch" not in missing:
        import torch
        if torch.cuda.is_available():
            device = f"cuda ({torch.cuda.get_device_name(0)})"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps (Apple GPU)"
        else:
            device = "cpu only: fine for data prep/inference, train on Colab"
        print(f"Device   : {device}")

    if "soundfile" not in missing:
        # Akan orthography check: ɛ and ɔ must survive a UTF-8 round trip.
        sample = "Ɛyɛ me dɛ sɛ mehunu wo. Ɔkɔɔ fie."
        assert sample.encode("utf-8").decode("utf-8") == sample
        print("Unicode  : ɛ / ɔ round-trip ok")

    if missing:
        print(f"\nInstall the missing packages: {', '.join(missing)}")
        return 1
    print("\nEnvironment ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
