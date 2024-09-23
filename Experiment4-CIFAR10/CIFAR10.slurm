#!/bin/bash

#SBATCH --job-name=ml_models_with_cifar10_data
#SBATCH --output=output_cifar10_%j.txt
#SBATCH --error=error_cifar10_%j.txt
#SBATCH --time=160:00:00
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=guptaa23@msu.edu

# Define models and category targets
models=("SVM" "RF" "GBM" "MLP" "CNN" "RNN")
category_targets=$(seq 0 9)
replicates=$(seq 1 5)
generation_interval=100  # Or whatever interval you choose

# Create output directories
mkdir -p output
mkdir -p error

# Submit jobs
for model in "${models[@]}"; do
  for category in $category_targets; do
    for replicate in $replicates; do
      if [[ "$model" == "CNN" || "$model" == "RNN" ]]; then
        partition="gpu"
        gres="--gres=gpu:1"
      else
        partition="compute"
        gres=""
      fi

      sbatch --partition=$partition $gres --export=ALL,MODEL_NAME=$model,CATEGORY_TARGET=$category,GEN_INTERVAL=$generation_interval,REPLICATE=$replicate <<EOF
#!/bin/bash

#SBATCH --job-name=cifar10_${model}_${category}_rep${replicate}
#SBATCH --output=output/cifar10_${model}_${category}_rep${replicate}_%j.txt
#SBATCH --error=error/cifar10_${model}_${category}_rep${replicate}_%j.txt
#SBATCH --time=160:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Load modules if necessary
# module load python/3.8.0
# module load cuda/10.1

# Activate the virtual environment
source ~/envs/myenv/bin/activate

# Execute the script
python CIFAR10.py \$MODEL_NAME \$CATEGORY_TARGET \$GEN_INTERVAL \$REPLICATE

# Deactivate the virtual environment after the job completes
deactivate
EOF
    done
  done
done