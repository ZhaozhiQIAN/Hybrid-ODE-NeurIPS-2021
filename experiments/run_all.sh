cd "$(dirname "$0")/.."  # cd to repo root.

printf "\n=========================== Runnung: create_data.sh ============================\n\n"
bash experiments/create_data.sh

printf "\n=============================== Runnung: Fig3.sh ===============================\n\n"
bash experiments/Fig3.sh

printf "\n============================= Runnung: run_dim.sh ==============================\n\n"
bash experiments/run_dim.sh

printf "\n========================= Runnung: run_noise_level.sh ==========================\n\n"
bash experiments/run_noise_level.sh

printf "\n============================= Runnung: Fig6.sh 10 ==============================\n\n"
bash experiments/Fig6.sh 10

printf "\n============================= Runnung: Fig6.sh 12 ==============================\n\n"
bash experiments/Fig6.sh 12

printf "\n=============================== Runnung: Fig7.sh ===============================\n\n"
bash experiments/Fig7.sh

printf "\n=============================== Runnung: Fig9.sh ===============================\n\n"
bash experiments/Fig9.sh
