
########################### high dim latents #####################################

device=c
sample=400
sample_total=`expr ${sample} + 800`
sample_cali=`expr ${sample} - 300`

# dim 8
echo "Running dim 8"
dim=8
model_path="model/model_dim${dim}/"
data_path="data/datafile_dim${dim}.pkl"

method=expert
python -u run_simulation.py --method=${method} --device=${device} --sample=${sample_total} --path=${model_path} --batch_size=10 --data_path=${data_path} --data_config="dim${dim}" > "results/dim${dim}_${method}.txt" &

method=hybrid
python -u run_simulation.py --method=${method} --device=${device} --sample=${sample_total} --path=${model_path} --batch_size=10 --arg_itr=1000 --restart=1 --data_path=${data_path} --data_config="dim${dim}" > "results/dim${dim}_${method}.txt" &

method=neural
python -u run_simulation.py --method=${method} --device=${device} --sample=${sample_total} --path=${model_path} --batch_size=10 --data_path=${data_path} --data_config="dim${dim}" > "results/dim${dim}_${method}.txt"

echo "Ensemble"
python -u run_simulation_residual.py --method=residual --device=${device} --sample=${sample_cali} --path=${model_path} --data_path=${data_path} --data_config="dim${dim}" > "results/dim${dim}_residual.txt" 
python -u run_simulation_ensemble.py --method=ensemble --device=${device} --sample=${sample_cali} --path=${model_path} --data_path=${data_path} --data_config="dim${dim}" > "results/dim${dim}_ensemble2.txt" 

# dim 12
echo "Running dim 12"
dim=12
model_path="model/model_dim${dim}/"
data_path="data/datafile_dim${dim}.pkl"

method=expert
python -u run_simulation.py --method=${method} --device=${device} --sample=${sample_total} --path=${model_path} --batch_size=10 --data_path=${data_path} --data_config="dim${dim}" > "results/dim${dim}_${method}.txt"

method=hybrid
python -u run_simulation.py --method=${method} --device=${device} --sample=${sample_total} --path=${model_path} --batch_size=10 --arg_itr=1000 --restart=1 --data_path=${data_path} --data_config="dim${dim}" > "results/dim${dim}_${method}.txt"

method=neural
python -u run_simulation.py --method=${method} --device=${device} --sample=${sample_total} --path=${model_path} --batch_size=100 --data_path=${data_path} --data_config="dim${dim}" > "results/dim${dim}_${method}.txt"

echo "Ensemble"
python -u run_simulation_residual.py --method=residual --device=${device} --sample=${sample_cali} --path=${model_path} --data_path=${data_path} --data_config="dim${dim}" > "results/dim${dim}_residual.txt"
python -u run_simulation_ensemble.py --method=ensemble --device=${device} --sample=${sample_cali} --path=${model_path} --data_path=${data_path} --data_config="dim${dim}" > "results/dim${dim}_ensemble2.txt" 

## todo evaluate t0

#t0=2
#
#
#model_arr=( hybrid neural expert )
#dim_arr=( 8 12 )
#
#for dim in "${dim_arr[@]}"
#do
#    for m in "${model_arr[@]}"
#    do
#        data_path="data/datafile_dim${dim}.pkl"
#        model_path="model/model_dim${dim}/"
#        python -u run_simulation.py --method=${m} --device=${device} --sample=${sample} --path=${model_path} --data_path=${data_path} --eval=y --t0=${t0} --data_config="dim${dim}" > "results/dim${dim}_${m}_${t0}.txt" &
#    done
#done
#
#echo "Ensemble"
#for dim in "${dim_arr[@]}"
#do
#    data_path="data/datafile_dim${dim}.pkl"
#    model_path="model/model_dim${dim}/"
#    python -u run_simulation_residual.py --method=residual --device=${device} --sample=${sample_cali} --path=${model_path} --data_path=${data_path} --data_config="dim${dim}" --eval=y --t0=${t0} > "results/dim${dim}_residual_${t0}.txt" &
#    python -u run_simulation_ensemble.py --method=ensemble --device=${device} --sample=${sample_cali} --path=${model_path} --data_path=${data_path} --data_config="dim${dim}" --eval=y --t0=${t0} > "results/dim${dim}_ensemble2_${t0}.txt"  &
#done

########################### summarize results #####################################

model_arr=( neural hybrid expert residual ensemble2 )
dim_arr=( 8 12 )

rm results/results_dim.txt

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

# t0 = 10
#t0=2
#
#model_arr=( neural hybrid expert residual ensemble2 )
#dim_arr=( 8 12 )
#
#rm results/results_dim_${t0}.txt
#
#for method in "${model_arr[@]}"
#do
#    for dim in "${dim_arr[@]}"
#    do
#        value=`tail -n 4 results/dim${dim}_${method}_${t0}.txt`
#        readarray -t y <<<"$value"
#        for line in "${y[@]}"
#        do
#            echo "${method},${dim},${line}" >> results/results_dim_${t0}.txt
#        done
#    done
#done
#
#for method in "${model_arr[@]}"
#do
#    value=`tail -n 4 results/sample_400_${method}_${t0}.txt`
#    readarray -t y <<<"$value"
#    for line in "${y[@]}"
#    do
#        echo "${method},6,${line}" >> results/results_dim_${t0}.txt
#    done
#done
#
#grep rmse_x results/results_dim.txt