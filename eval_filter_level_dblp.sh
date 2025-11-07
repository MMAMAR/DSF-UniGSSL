python eval_filter_level.py --dataset=dblp --pretraining_type=node --finetuning_type=prox --device_number=0
python eval_filter_level.py --dataset=dblp --pretraining_type=node --finetuning_type=cluster --device_number=0
python eval_filter_level.py --dataset=dblp --pretraining_type=node --finetuning_type=graph --device_number=0

python eval_filter_level.py --dataset=dblp --pretraining_type=prox --finetuning_type=cluster --device_number=0
python eval_filter_level.py --dataset=dblp --pretraining_type=prox --finetuning_type=graph --device_number=0
python eval_filter_level.py --dataset=dblp --pretraining_type=cluster --finetuning_type=graph --device_number=0