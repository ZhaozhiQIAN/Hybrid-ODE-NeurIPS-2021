cd "$(dirname "$0")/.."  # cd to repo root.



######## hidden layer for NODE

device=c  # For CPU set device=c; for CUDA set device=0 (if 0 is your the CUDA device number)
data_path=data/datafile_dose_exp_test.pkl

sample=400
sample_total=`expr ${sample} + 800`

encoder_output_dim_arr=( 10 15 )

for encoder_output_dim in "${encoder_output_dim_arr[@]}"
do
    model_path="model/model_sample_${sample}/Z_${encoder_output_dim}"
    python -u -m experiments.run_simulation --method=neural --encoder_output_dim=$encoder_output_dim --device=${device} --batch_size=10 --sample=${sample_total} --data_path=${data_path} --path=${model_path} > "results/sample_${sample}_Z_${encoder_output_dim}.txt"
done

rm -f results/results_z.txt

value=`tail -n 4 "results/sample_${sample}_neural.txt"`
readarray -t y <<<"$value"
for line in "${y[@]}"
do
    echo "neural,6,${line}" >> results/results_z.txt
done

seed_arr=( 10 15 )
for seed in "${seed_arr[@]}"
do
    value=`tail -n 4 "results/sample_${sample}_Z_${encoder_output_dim}.txt"`
    readarray -t y <<<"$value"
    for line in "${y[@]}"
    do
        echo "neural,${seed},${line}" >> results/results_z.txt
    done
done
