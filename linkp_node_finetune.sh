
python linkp_node.py --dataset=cora --pretrained=True --pretrained_type=cluster --device_number=0
python linkp_node.py --dataset=cora --pretrained=True --pretrained_type=graph --device_number=0
python linkp_node.py --dataset=cora --pretrained=True --pretrained_type=prox --device_number=0

python linkp_node.py --dataset=dblp --pretrained=True --pretrained_type=cluster --batch_size=1024 --device_number=0
python linkp_node.py --dataset=dblp --pretrained=True --pretrained_type=graph --batch_size=1024 --device_number=0
python linkp_node.py --dataset=dblp --pretrained=True --pretrained_type=prox --batch_size=1024 --device_number=0
