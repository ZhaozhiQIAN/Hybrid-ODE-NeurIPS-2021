mkdir results
mkdir model
device=c


############################## New dataset dose_exp ###############################


echo "Hybrid"
sample=310
model_path="model/model_sample_${sample}/"
python -u run_simulation.py --method=hybrid --device=${device} --sample=${sample} --path=${model_path} --batch_size=10 > "results/sample_${sample}_h.txt"

sample_arr=( 400 800 )

for sample in "${sample_arr[@]}"
do
    model_path="model/model_sample_${sample}/"
    python -u run_simulation.py --method=hybrid --device=${device} --sample=${sample} --path=${model_path} --batch_size=10 > "results/sample_${sample}_h.txt"
done


# neural
echo "neural"
sample=310
model_path="model/model_sample_${sample}/"
python -u run_simulation.py --method=neural --device=${device} --sample=${sample} --path=${model_path} --batch_size=10 > "results/sample_${sample}_n.txt"

sample=400
model_path="model/model_sample_${sample}/"
python -u run_simulation.py --method=neural --device=${device} --sample=${sample} --path=${model_path} --batch_size=10 > "results/sample_${sample}_n.txt"

sample=800
model_path="model/model_sample_${sample}/"
python -u run_simulation.py --method=neural --device=${device} --sample=${sample} --path=${model_path} --batch_size=10 > "results/sample_${sample}_n.txt"


# expert

echo "expert"
sample=310
model_path="model/model_sample_${sample}/"
python -u run_simulation.py --method=expert --device=${device} --sample=${sample} --path=${model_path} --batch_size=10 > "results/sample_${sample}_e.txt"


sample_arr=( 400 800 )

for sample in "${sample_arr[@]}"
do
    model_path="model/model_sample_${sample}/"
    python -u run_simulation.py --method=expert --device=${device} --sample=${sample} --path=${model_path} --batch_size=10 > "results/sample_${sample}_e.txt"
done


######################## Evaluation #################################

data_path=data/datafile_dose_exp_test.pkl

model_arr=( hybrid neural expert )
sample_arr=( 310 400 800 )

for sample in "${sample_arr[@]}"
do
    for m in "${model_arr[@]}"
    do
        model_path="model/model_sample_${sample}/"
        python -u run_simulation.py --method=${m} --device=${device} --sample=${sample} --path=${model_path} --data_path=${data_path} --eval=y > "results/sample_${sample}_${m}.txt" &
    done
done


for sample in "${sample_arr[@]}"
do
    model_path="model/model_sample_${sample}/"
    sample_cali=`expr ${sample} - 300`
    python -u run_simulation_ensemble.py --method=ensemble --device=${device} --sample=${sample_cali} --path=${model_path} --data_path=${data_path} > "results/sample_${sample}_ensemble2.txt"
    python -u run_simulation_residual.py --method=residual --device=${device} --sample=${sample_cali} --path=${model_path} --data_path=${data_path} > "results/sample_${sample}_residual.txt"
done


######################## Summary #################################


model_arr=( neural expert hybrid residual ensemble2 )
seed_arr=( 310 400 800 )


rm results/results_sample.txt
for seed in "${seed_arr[@]}"
do
    for m in "${model_arr[@]}"
    do
        value=`tail -n 4 results/sample_${seed}_${m}.txt`
        readarray -t y <<<"$value"
        for line in "${y[@]}"
        do
            echo "${m},${seed},${line}" >> results/results_sample.txt
        done
    done
done

grep rmse_x results/results_sample.txt

