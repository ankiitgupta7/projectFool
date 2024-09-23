#!/bin/bash

#SBATCH --job-name=ml_models_with_fashion_mnist_data
#SBATCH --output=output_fashion_mnist_%j.txt
#SBATCH --error=error_fashion_mnist_%j.txt
#SBATCH --time=160:00:00
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=guptaa23@msu.edu

# Define models and category targets
models=("SVM" "RF" "GBM" "MLP" "CNN" "RNN")
category_targets=$(seq 0 9)
replicates=$(seq 1 3)
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

#SBATCH --job-name=fashion_mnist_${model}_${category}_rep${replicate}
#SBATCH --output=output/fashion_mnist_${model}_${category}_rep${replicate}_%j.txt
#SBATCH --error=error/fashion_mnist_${model}_${category}_rep${replicate}_%j.txt
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Activate the virtual environment
source ~/envs/myenv/bin/activate

# Execute the script
python mnistFashion.py \$MODEL_NAME \$CATEGORY_TARGET \$GEN_INTERVAL \$REPLICATE

# Deactivate the virtual environment after the job completes
deactivate
EOF
    done
  done
done
