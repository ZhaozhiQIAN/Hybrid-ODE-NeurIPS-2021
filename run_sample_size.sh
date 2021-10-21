
device=c


############################## New dataset dose_exp ###############################

# sparsity in ODE and outputs
# noise 0.2
# kel 1
# dose [0. 10]
# t_0 test 5

echo "Hybrid"
sample=310
model_path="model/model_sample_${sample}/"
python -u run_simulation.py --method=hybrid --device=${device} --sample=${sample} --path=${model_path} --batch_size=10 --seed=1 > "results/sample_${sample}_h.txt"

sample=400
model_path="model/model_sample_${sample}/"
python -u run_simulation.py --method=hybrid --device=${device} --sample=${sample} --path=${model_path} > "results/sample_${sample}_h.txt" 

sample=1300
old_sample=400
model_path="model/model_sample_${sample}/"
init_path="model/model_sample_${old_sample}/"
python -u run_simulation.py --method=hybrid --device=${device} --sample=${sample} --init=${init_path} --path=${model_path} > "results/sample_${sample}_h.txt"


sample=800
old_sample=400
model_path="model/model_sample_${sample}/"
init_path="model/model_sample_${old_sample}/"
python -u run_simulation.py --method=hybrid --device=${device} --sample=${sample} --init=${init_path} --path=${model_path} --seed=10 > "results/sample_${sample}_h.txt" &


# neural
echo "neural"
sample=310
model_path="model/model_sample_${sample}/"
python -u run_simulation.py --method=neural --device=${device} --sample=${sample} --path=${model_path} --batch_size=10 > "results/sample_${sample}_n.txt"

sample=400
model_path="model/model_sample_${sample}/"
python -u run_simulation.py --method=neural --device=${device} --sample=${sample} --path=${model_path} --seed=4 > "results/sample_${sample}_n.txt"

sample=1300
old_sample=400
model_path="model/model_sample_${sample}/"
init_path="model/model_sample_${old_sample}/"
python -u run_simulation.py --method=neural --device=${device} --sample=${sample} --init=${init_path} --path=${model_path} --seed=5 > "results/sample_${sample}_n.txt"

sample=800
old_sample=400
model_path="model/model_sample_${sample}/"
init_path="model/model_sample_${old_sample}/"
python -u run_simulation.py --method=neural --device=${device} --sample=${sample} --init=${init_path} --path=${model_path} --seed=7 > "results/sample_${sample}_n.txt" &


# expert
echo "expert"
sample=310
model_path="model/model_sample_${sample}/"
python -u run_simulation.py --method=expert --device=${device} --sample=${sample} --path=${model_path} --batch_size=10 > "results/sample_${sample}_e.txt"

sample=400
old_sample=310
model_path="model/model_sample_${sample}/"
init_path="model/model_sample_${old_sample}/"
python -u run_simulation.py --method=expert --device=${device} --sample=${sample} --path=${model_path} --init=${init_path} --seed=5 > "results/sample_${sample}_e.txt"


sample=1300
old_sample=310
model_path="model/model_sample_${sample}/"
init_path="model/model_sample_${old_sample}/"
python -u run_simulation.py --method=expert --device=${device} --sample=${sample} --init=${init_path} --path=${model_path} > "results/sample_${sample}_e.txt"  &

sample=800
old_sample=310
model_path="model/model_sample_${sample}/"
init_path="model/model_sample_${old_sample}/"
python -u run_simulation.py --method=expert --device=${device} --sample=${sample} --init=${init_path} --path=${model_path} > "results/sample_${sample}_e.txt"  &

######################## Evaluation #################################

data_path=data/datafile_dose_exp_test.pkl

model_arr=( hybrid neural expert )
sample_arr=( 310 400 800 1300 )

for sample in "${sample_arr[@]}"
do
    for m in "${model_arr[@]}"
    do
        model_path="model/model_sample_${sample}/"
        python -u run_simulation.py --method=${m} --device=${device} --sample=${sample} --path=${model_path} --data_path=${data_path} --eval=y > "results/sample_${sample}_${m}.txt"
    done
done

# todo: these are updated
#sample=1300
#m=neural
#model_path="model/model_sample_${sample}/"
#python -u run_simulation.py --method=${m} --device=${device} --sample=${sample} --path=${model_path} --data_path=${data_path} --eval=y > "results/sample_${sample}_${m}.txt"
#sample=400
#m=neural
#model_path="model/model_sample_${sample}/"
#python -u run_simulation.py --method=${m} --device=${device} --sample=${sample} --path=${model_path} --data_path=${data_path} --eval=y > "results/sample_${sample}_${m}.txt"
#
#sample=400
#m=expert
#model_path="model/model_sample_${sample}/"
#python -u run_simulation.py --method=${m} --device=${device} --sample=${sample} --path=${model_path} --data_path=${data_path} --eval=y > "results/sample_${sample}_${m}.txt" &
#
#sample=310
#m=hybrid
#model_path="model/model_sample_${sample}/"
#python -u run_simulation.py --method=${m} --device=${device} --sample=${sample} --path=${model_path} --data_path=${data_path} --eval=y > "results/sample_${sample}_${m}.txt" &

sample_arr=( 800 )


for sample in "${sample_arr[@]}"
do
    model_path="model/model_sample_${sample}/"
    sample_cali=`expr ${sample} - 300`
    python -u run_simulation_ensemble.py --method=ensemble --device=${device} --sample=${sample_cali} --path=${model_path} --data_path=${data_path} > "results/sample_${sample}_ensemble2.txt" &
    python -u run_simulation_residual.py --method=residual --device=${device} --sample=${sample_cali} --path=${model_path} --data_path=${data_path} > "results/sample_${sample}_residual.txt" &
done

sample=310
model_path="model/model_sample_${sample}/"
sample_cali=`expr ${sample} - 300`
nohup python -u run_simulation_residual.py --method=residual --device=${device} --sample=${sample_cali} --path=${model_path} --data_path=${data_path} --seed=4 > "results/sample_${sample}_residual.txt" &


#python -u run_simulation_ensemble.py --method=ensemble --device=${device} --sample=${sample_cali} --path=${model_path} --data_path=${data_path} > "results/sample_${sample}_ensemble2.txt"

# t0=10

data_path=data/datafile_dose_exp_test.pkl

model_arr=( hybrid neural expert )
sample_arr=( 800 )
t0=10

for sample in "${sample_arr[@]}"
do
    for m in "${model_arr[@]}"
    do
        model_path="model/model_sample_${sample}/"
        python -u run_simulation.py --method=${m} --device=${device} --sample=${sample} --path=${model_path} --data_path=${data_path} --eval=y --t0=${t0} > "results/sample_${sample}_${m}_${t0}.txt" &
    done
done


for sample in "${sample_arr[@]}"
do
    model_path="model/model_sample_${sample}/"
    sample_cali=`expr ${sample} - 300`
    nohup python -u run_simulation_ensemble.py --t0=${t0} --method=ensemble --device=${device} --sample=${sample_cali} --path=${model_path} --data_path=${data_path} > "results/sample_${sample}_ensemble2_${t0}.txt" &
    nohup python -u run_simulation_residual.py --t0=${t0} --eval=y --method=residual --device=${device} --sample=${sample_cali} --path=${model_path} --data_path=${data_path} > "results/sample_${sample}_residual_${t0}.txt" &
done


######################## Summary #################################


model_arr=( neural expert hybrid residual ensemble2 )
seed_arr=( 310 400 800 1300 )


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


# 10
model_arr=( neural expert hybrid residual ensemble2 )
seed_arr=( 310 400 800 1300 )
t0=10


rm results/results_sample_${t0}.txt
for seed in "${seed_arr[@]}"
do
    for m in "${model_arr[@]}"
    do
        value=`tail -n 4 results/sample_${seed}_${m}_${t0}.txt`
        readarray -t y <<<"$value"
        for line in "${y[@]}"
        do
            echo "${m},${seed},${line}" >> results/results_sample_${t0}.txt
        done
    done
done

grep rmse_x results/results_sample_${t0}.txt



############################## rebuttal ablation ###############################


echo "Hybrid"
sample=310
model_path="model/model_sample_${sample}/"
python -u run_simulation.py --method=hybrid --device=${device} --sample=${sample} --path=${model_path} --ablate=True --batch_size=10 --restart=1 > "results/sample_${sample}_a.txt" &

sample=400
model_path="model/model_sample_${sample}/"
data_path=data/datafile_dose_exp_test.pkl

python -u run_simulation.py --method=hybrid --device=${device} --sample=${sample} --path=${model_path} --ablate=True --restart=1 > "results/sample_${sample}_a.txt" &

#seed_arr=( 1 2 3 4 5 )
#
#for seed in "${seed_arr[@]}"
#do
#  python -u run_simulation.py --method=hybrid --device=${device} --sample=${sample} --path=${model_path} --ablate=True --restart=1 --seed=${seed} > "results/sample_${sample}_a.txt" &
#done

sample=1300
old_sample=400
model_path="model/model_sample_${sample}/"
#init_path="model/model_sample_${old_sample}/"
#--init=${init_path}
python -u run_simulation.py --method=hybrid --device=${device} --sample=${sample}  --path=${model_path} --ablate=True --restart=1 > "results/sample_${sample}_a.txt" &


sample=800
old_sample=400
model_path="model/model_sample_${sample}/"
#init_path="model/model_sample_${old_sample}/"
#--init=${init_path}
python -u run_simulation.py --method=hybrid --device=${device} --sample=${sample} --path=${model_path} --ablate=True --restart=1 > "results/sample_${sample}_a.txt" &


# eval
sample_arr=( 310 400 800 1300 )
data_path=data/datafile_dose_exp_test.pkl

for sample in "${sample_arr[@]}"
do
    method=hybrid
    model_path="model/model_sample_${sample}/"
    python -u run_simulation.py --method=${method} --device=${device} --sample=${sample} --path=${model_path} --data_path=${data_path} --ablate=True --eval=y > "results/sample_${sample}_a_eval.txt" &
done
