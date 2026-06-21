# Remaining seeds
$seeds = @(42,43,44,45,46)

# Gates for ablation (ONLY adapters)
$gates = @(0.0,0.25,0.5,0.75,1.0)

foreach ($seed in $seeds) {

    Write-Host ""
    Write-Host "======================================="
    Write-Host "Starting Seed $seed"
    Write-Host "======================================="
    Write-Host ""

    # -----------------------------
    # 1. LINEAR BASE (FIXED NONLINEARITY)
    # -----------------------------
    Write-Host ""
    Write-Host "Linear Base | Gate=$gate | Seed=$seed"
    Write-Host ""

    python train_LoRA_Qudratic_relu_ablation.py `
        --dataset cifar10 `
        --stage linear_base `
        --epochs 10 `
        --relu-gate $seed `
        --seed $seed


    foreach ($gate in $gates) {

        Write-Host ""
        Write-Host "Quadratic Adapter | Gate=$gate | Seed=$seed"
        Write-Host ""

        python train_LoRA_Qudratic_relu_ablation.py `
            --dataset cifar10 `
            --stage quadratic_adapter `
            --base-checkpoint "outputs/cifar10/linear_base/relu_gate_1.0/seed_$seed/model.pt" `
            --quad-rank 4 `
            --epochs 10 `
            --relu-gate $gate `
            --seed $seed


        Write-Host ""
        Write-Host "LoRA Adapter | Gate=$gate | Seed=$seed"
        Write-Host ""

        python train_LoRA_Qudratic_relu_ablation.py `
            --dataset cifar10 `
            --stage lora_adapter `
            --base-checkpoint "outputs/cifar10/linear_base/relu_gate_1.0/seed_$seed/model.pt" `
            --quad-rank 4 `
            --epochs 10 `
            --relu-gate $gate `
            --seed $seed
    }
}

Write-Host ""
Write-Host "======================================="
Write-Host "ALL EXPERIMENTS FINISHED"
Write-Host "======================================="