device=c


t0="$1"
#t0=12
#t0=10
data_path=data/datafile_dose_exp_test.pkl

model_arr=( hybrid neural expert )
sample_arr=( 310 400 800 )


for sample in "${sample_arr[@]}"
do
    for m in "${model_arr[@]}"
    do
        model_path="model/model_sample_${sample}/"
        python -u run_simulation.py --method=${m} --device=${device} --sample=${sample} --path=${model_path} --data_path=${data_path} --eval=y --t0=${t0} > "results/sample_${sample}_${m}_${t0}.txt"
    done
done


for sample in "${sample_arr[@]}"
do
    model_path="model/model_sample_${sample}/"
    sample_cali=`expr ${sample} - 300`
    python -u run_simulation_ensemble.py --t0=${t0} --method=ensemble --device=${device} --sample=${sample_cali} --path=${model_path} --data_path=${data_path} > "results/sample_${sample}_ensemble2_${t0}.txt"
    python -u run_simulation_residual.py --t0=${t0} --eval=y --method=residual --device=${device} --sample=${sample_cali} --path=${model_path} --data_path=${data_path} > "results/sample_${sample}_residual_${t0}.txt"
done


model_arr=( neural expert hybrid residual ensemble2 )

rm results/results_sample_${t0}.txt
for sample in "${sample_arr[@]}"
do
    for m in "${model_arr[@]}"
    do
        value=`tail -n 4 results/sample_${sample}_${m}_${t0}.txt`
        readarray -t y <<<"$value"
        for line in "${y[@]}"
        do
            echo "${m},${sample},${line}" >> results/results_sample_${t0}.txt
        done
    done
done

grep rmse_x results/results_sample_${t0}.txt




