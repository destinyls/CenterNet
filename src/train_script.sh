python main.py ddd --arch res_18 --batch_size 8 --master_batch 7 --num_epochs 70 --lr_step 45,60 --checkpoints_path "/root/CenterNet/checkpoints/RESNET18_FPN_002nd"
python evaluate.py ddd --arch res_18 --checkpoints_path "/root/CenterNet/checkpoints/RESNET18_FPN_002nd"

python main.py ddd --arch res_18 --batch_size 8 --master_batch 7 --num_epochs 70 --lr_step 45,60 --checkpoints_path "/root/CenterNet/checkpoints/RESNET18_FPN_003nd"
python evaluate.py ddd --arch res_18 --checkpoints_path "/root/CenterNet/checkpoints/RESNET18_FPN_003nd"

python main.py ddd --arch res_18 --batch_size 8 --master_batch 7 --num_epochs 70 --lr_step 45,60 --checkpoints_path "/root/CenterNet/checkpoints/RESNET18_FPN_004nd"
python evaluate.py ddd --arch res_18 --checkpoints_path "/root/CenterNet/checkpoints/RESNET18_FPN_004nd"

python main.py ddd --arch res_18 --batch_size 8 --master_batch 7 --num_epochs 70 --lr_step 45,60 --checkpoints_path "/root/CenterNet/checkpoints/RESNET18_FPN_005nd"
python evaluate.py ddd --arch res_18 --checkpoints_path "/root/CenterNet/checkpoints/RESNET18_FPN_005nd"
