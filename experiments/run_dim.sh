cd "$(dirname "$0")/.."  # cd to repo root.



########################### high dim latents #####################################

device=c  # For CPU set device=c; for CUDA set device=0 (if 0 is your the CUDA device number)
sample=400
sample_total=`expr ${sample} + 800`
sample_cali=`expr ${sample} - 300`

# dim 8
echo "Running dim 8"
dim=8
model_path="model/model_dim${dim}/"
data_path="data/datafile_dim${dim}.pkl"

method=expert
python -u -m experiments.run_simulation --method=${method} --device=${device} --sample=${sample_total} --path=${model_path} --batch_size=10 --data_path=${data_path} --data_config="dim${dim}" > "results/dim${dim}_${method}.txt" &

method=hybrid
python -u -m experiments.run_simulation --method=${method} --device=${device} --sample=${sample_total} --path=${model_path} --batch_size=10 --arg_itr=1000 --restart=1 --data_path=${data_path} --data_config="dim${dim}" > "results/dim${dim}_${method}.txt"

method=neural
python -u -m experiments.run_simulation --method=${method} --device=${device} --sample=${sample_total} --path=${model_path} --batch_size=10 --data_path=${data_path} --data_config="dim${dim}" > "results/dim${dim}_${method}.txt"

echo "Ensemble"
python -u -m experiments.run_simulation_residual --method=residual --device=${device} --sample=${sample_cali} --path=${model_path} --data_path=${data_path} --data_config="dim${dim}" > "results/dim${dim}_residual.txt" 
python -u -m experiments.run_simulation_ensemble --method=ensemble --device=${device} --sample=${sample_cali} --path=${model_path} --data_path=${data_path} --data_config="dim${dim}" > "results/dim${dim}_ensemble2.txt" 

# dim 12
echo "Running dim 12"
dim=12
model_path="model/model_dim${dim}/"
data_path="data/datafile_dim${dim}.pkl"

method=expert
python -u -m experiments.run_simulation --method=${method} --device=${device} --sample=${sample_total} --path=${model_path} --batch_size=10 --data_path=${data_path} --data_config="dim${dim}" > "results/dim${dim}_${method}.txt" &

method=hybrid
python -u -m experiments.run_simulation --method=${method} --device=${device} --sample=${sample_total} --path=${model_path} --batch_size=10 --arg_itr=1000 --restart=1 --data_path=${data_path} --data_config="dim${dim}" > "results/dim${dim}_${method}.txt"

method=neural
python -u -m experiments.run_simulation --method=${method} --device=${device} --sample=${sample_total} --path=${model_path} --batch_size=100 --data_path=${data_path} --data_config="dim${dim}" > "results/dim${dim}_${method}.txt"

echo "Ensemble"
python -u -m experiments.run_simulation_residual --method=residual --device=${device} --sample=${sample_cali} --path=${model_path} --data_path=${data_path} --data_config="dim${dim}" > "results/dim${dim}_residual.txt"
python -u -m experiments.run_simulation_ensemble --method=ensemble --device=${device} --sample=${sample_cali} --path=${model_path} --data_path=${data_path} --data_config="dim${dim}" > "results/dim${dim}_ensemble2.txt" 



########################### summarize results #####################################

model_arr=( neural hybrid expert residual ensemble2 )
dim_arr=( 8 12 )

rm -f results/results_dim.txt

for method in "${model_arr[@]}"
do
    for dim in "${dim_arr[@]}"
    do
        value=`tail -n 4 results/dim${dim}_${method}.txt`
        readarray -t y <<<"$value"
        for line in "${y[@]}"
        do
            echo "${method},${dim},${line}" >> results/results_dim.txt
        done
    done
done

for method in "${model_arr[@]}"
do
    value=`tail -n 4 results/sample_400_${method}.txt`
    readarray -t y <<<"$value"
    for line in "${y[@]}"
    do
        echo "${method},6,${line}" >> results/results_dim.txt
    done
done

grep rmse_x results/results_dim.txt
