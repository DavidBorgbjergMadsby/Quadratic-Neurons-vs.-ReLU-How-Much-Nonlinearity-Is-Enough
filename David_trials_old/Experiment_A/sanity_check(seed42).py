from pathlib import Path

BASE_DIR = Path("outputs/cifar10/linear_base")
SEEDS = [43, 44, 45, 46]


def check_linear_base():
    print("\n🔍 Checking linear_base checkpoints...\n")

    missing = []

    for seed in SEEDS:
        path = BASE_DIR / f"seed_{seed}" / "model.pt"

        if path.exists():
            print(f"✔ Found: {path}")
        else:
            print(f"❌ Missing: {path}")
            missing.append(str(path))

    return missing


def check_for_bad_gate_in_base():
    print("\n🔍 Checking for illegal gate folders in linear_base...\n")

    bad = list(BASE_DIR.glob("**/relu_gate_*"))

    if len(bad) == 0:
        print("✔ No gate folders inside linear_base (good)")
    else:
        for b in bad:
            print(f"⚠️ Unexpected gate folder: {b}")

    return bad


def check_adapter_structure(stage):
    print(f"\n🔍 Checking {stage} structure...\n")

    base = Path("outputs/cifar10") / stage

    if not base.exists():
        print(f"❌ Missing stage folder: {base}")
        return [base]

    issues = []

    for gate_dir in base.glob("relu_gate_*"):
        for seed in SEEDS:
            path = gate_dir / f"seed_{seed}" / "model.pt"

            if path.exists():
                print(f"✔ {path}")
            else:
                print(f"❌ Missing: {path}")
                issues.append(path)

    return issues


def main():
    print("\n===================================")
    print("ABALATION SANITY CHECK")
    print("===================================\n")

    issues = []

    issues += check_linear_base()
    check_for_bad_gate_in_base()

    issues += check_adapter_structure("quadratic_adapter")
    issues += check_adapter_structure("lora_adapter")

    print("\n===================================")

    if len(issues) == 0:
        print("✅ ALL CHECKS PASSED — SAFE TO RUN EXPERIMENTS")
    else:
        print(f"❌ FOUND {len(issues)} ISSUES")
        print("Fix these before running PS1")

    print("===================================\n")


if __name__ == "__main__":
    main()
