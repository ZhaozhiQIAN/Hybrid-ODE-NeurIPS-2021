device=c


t0=12
data_path=data/datafile_dose_exp_test.pkl

model_arr=( hybrid neural expert )
sample_arr=( 310 400 800 )


for sample in "${sample_arr[@]}"
do
    for m in "${model_arr[@]}"
    do
        model_path="model/model_sample_${sample}/"
        result_path="results/sample_${sample}_${m}_${t0}.pkl"
        python -u run_eval.py --method=${m} --device=${device} --sample=${sample} --path=${model_path} --result_path=${result_path} --data_path=${data_path} --eval=y --t0=${t0}
    done
done


for sample in "${sample_arr[@]}"
do
    model_path="model/model_sample_${sample}/"
    sample_cali=`expr ${sample} - 300`
    python -u run_simulation_ensemble.py --t0=${t0} --method=ensemble --device=${device} --sample=${sample_cali} --path=${model_path} --data_path=${data_path} --result_path="results/sample_${sample}_ensemble_${t0}.pkl" --horizon=True
    python -u run_simulation_residual.py --t0=${t0} --eval=y --method=residual --device=${device} --sample=${sample_cali} --path=${model_path} --data_path=${data_path} --result_path="results/sample_${sample}_residual_${t0}.pkl" --horizon=True
done

