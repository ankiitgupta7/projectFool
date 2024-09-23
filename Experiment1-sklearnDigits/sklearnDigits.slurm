#!/bin/bash

#SBATCH --job-name=ml_models_with_sklearn_data
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=guptaa23@msu.edu

# Define models and digit targets
models=("SVM" "RF" "GBM" "MLP" "CNN" "RNN")
digit_targets=$(seq 0 9)
replicates=$(seq 1 5)
generation_interval=10  # Or whatever interval you choose

# Create output directories
mkdir -p output
mkdir -p error

# Submit jobs
for model in "${models[@]}"; do
  for digit in $digit_targets; do
    for replicate in $replicates; do
      if [[ "$model" == "CNN" || "$model" == "RNN" ]]; then
        partition="gpu"
        gres="--gres=gpu:1"
      else
        partition="compute"
        gres=""
      fi

      sbatch --partition=$partition $gres --export=ALL,MODEL_NAME=$model,TARGET_DIGIT=$digit,GEN_INTERVAL=$generation_interval,REPLICATE=$replicate <<EOF
#!/bin/bash

#SBATCH --job-name=${model}_${digit}_rep${replicate}
#SBATCH --output=output/${model}_${digit}_rep${replicate}_%j.txt
#SBATCH --error=error/${model}_${digit}_rep${replicate}_%j.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Activate the persistent virtual environment
source ~/envs/myenv/bin/activate

# Execute the script
python sklearnDigits.py \$MODEL_NAME \$TARGET_DIGIT \$GEN_INTERVAL \$REPLICATE

# Deactivate the virtual environment after the job completes
deactivate
EOF
    done
  done
done
