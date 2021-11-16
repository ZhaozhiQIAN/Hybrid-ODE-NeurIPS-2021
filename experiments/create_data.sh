cd "$(dirname "$0")/.."  # cd to repo root.

mkdir -p data

python -u -m generated_data.generate_data_train
python -u -m generated_data.generate_data_test
python -u -m generated_data.generate_data_noise --noise_level=0.4
python -u -m generated_data.generate_data_noise --noise_level=0.8
python -u -m generated_data.generate_data_noise --noise_level=1.0
python -u -m generated_data.generate_data_dim8
python -u -m generated_data.generate_data_dim12
