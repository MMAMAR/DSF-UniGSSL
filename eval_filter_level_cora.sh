python eval_filter_level.py --dataset=cora --pretraining_type=node --finetuning_type=prox --filtering_threshold=4.0 --pca_n_comp=10 --device_number=0
python eval_filter_level.py --dataset=cora --pretraining_type=node --finetuning_type=cluster --device_number=0
python eval_filter_level.py --dataset=cora --pretraining_type=node --finetuning_type=graph --device_number=0

python eval_filter_level.py --dataset=cora --pretraining_type=prox --finetuning_type=cluster --device_number=0
python eval_filter_level.py --dataset=cora --pretraining_type=prox --finetuning_type=graph --device_number=0
python eval_filter_level.py --dataset=cora --pretraining_type=cluster --finetuning_type=graph --device_number=0