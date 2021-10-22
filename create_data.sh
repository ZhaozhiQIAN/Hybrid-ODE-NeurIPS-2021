mkdir data
python -u generate_data_train.py
python -u generate_data_test.py
python -u generate_data_noise.py --noise_level=0.4
python -u generate_data_noise.py --noise_level=0.8
python -u generate_data_noise.py --noise_level=1.0
python -u generate_data_dim8.py
python -u generate_data_dim12.py
