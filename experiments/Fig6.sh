cd "$(dirname "$0")/.."  # cd to repo root.

device=c  # For CPU set device=c; for CUDA set device=0 (if 0 is your the CUDA device number)



t0="$1"  # {12, 10}
data_path=data/datafile_dose_exp_test.pkl

model_arr=( hybrid neural expert )
sample_arr=( 310 400 800 )

for sample in "${sample_arr[@]}"
do
    for m in "${model_arr[@]}"
    do
        model_path="model/model_sample_${sample}/"
        python -u -m experiments.run_simulation --method=${m} --device=${device} --sample=${sample} --path=${model_path} --data_path=${data_path} --eval=y --t0=${t0} > "results/sample_${sample}_${m}_${t0}.txt"
    done
done

for sample in "${sample_arr[@]}"
do
    model_path="model/model_sample_${sample}/"
    sample_cali=`expr ${sample} - 300`
    python -u -m experiments.run_simulation_ensemble --t0=${t0} --method=ensemble --device=${device} --sample=${sample_cali} --path=${model_path} --data_path=${data_path} > "results/sample_${sample}_ensemble2_${t0}.txt"
    python -u -m experiments.run_simulation_residual --t0=${t0} --eval=y --method=residual --device=${device} --sample=${sample_cali} --path=${model_path} --data_path=${data_path} > "results/sample_${sample}_residual_${t0}.txt"
done

model_arr=( neural expert hybrid residual ensemble2 )

rm -f results/results_sample_${t0}.txt
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
