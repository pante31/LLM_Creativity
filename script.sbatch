#!/bin/bash
#SBATCH --job-name=LLM_Creativity         
#SBATCH --mail-type=ALL                  
#SBATCH --mail-user=alessandro.tutone@studenti.unibo.it
#SBATCH --time=20:00:00                   
#SBATCH --nodes=1                         
#SBATCH --ntasks-per-node=1               
#SBATCH --cpus-per-task=4                 
#SBATCH --mem=48G                         
#SBATCH --partition=l40
#SBATCH --output=/scratch.hpc/alessandro.tutone/LLM_creativity/output/%j.out
#SBATCH --error=/scratch.hpc/alessandro.tutone/LLM_creativity/output/%j.err
#SBATCH --chdir=/scratch.hpc/alessandro.tutone/LLM_creativity
#SBATCH --gres=gpu:1                  

source venv/bin/activate
pip3 install torch --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir
pip3 install -r requirements.txt --no-cache-dir
python -m spacy download en_core_web_sm
venv/bin/python3 creativity.py