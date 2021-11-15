
cd "$(dirname "$0")/.."  # cd to repo root.



sample_arr=( 100 250 500 1000 )
for sample in "${sample_arr[@]}"
do
    python -u -m experiments.run_real --sample=${sample} --method=neural --ode_method=midpoint > results/neural_${sample}.txt
    python -u -m experiments.run_real --sample=${sample} --method=2nd --ode_method=rk4 --encoder_output_dim=40 > results/2nd_${sample}.txt

    python -u -m experiments.run_real --sample=${sample} --method=tlstm > results/tlstm_${sample}.txt
    python -u -m experiments.run_real --sample=${sample} --method=gruode  > results/gruode_${sample}.txt

    python -u -m experiments.run_real --sample=${sample} --method=hybrid --ode_method=midpoint  > results/hybrid_${sample}.txt

    python -u -m experiments.run_real --sample=${sample} --method=expert --encoder_output_dim=4 --ode_method=midpoint  > results/expert_${sample}.txt

    python -u -m experiments.run_real_ensemble --sample=${sample} --method=ensemble --ode_method=midpoint  > results/ensemble_${sample}.txt

    python -u -m experiments.run_real_residual --sample=${sample} --method=residual --ode_method=midpoint  > results/residual_${sample}.txt &

done



######################## summarize results ###################################

model_arr=( neural 2nd tlstm gruode hybrid expert ensemble residual )

rm -f results/results_real_sample.csv

for m in "${model_arr[@]}"
do
for sample in "${sample_arr[@]}"
    do
        value=`tail -n 3 results/${m}_${sample}.txt`
        readarray -t y <<<"$value"
        for line in "${y[@]}"
        do
            echo "${m},${sample},${line}" >> results/results_real_sample.csv
        done
    done
done
