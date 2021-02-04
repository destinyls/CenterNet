python main.py ddd --arch dla_34 --batch_size 8 --val_intervals 2 --master_batch 7 --num_epochs 140 --lr_step 45,60 --checkpoints_path "/root/CenterNet/checkpoints/Resnet34_Light_FPN_Attention_Loss_001nd"
python evaluate.py ddd --arch dla_34 --checkpoints_path "/root/CenterNet/checkpoints/Resnet34_Light_FPN_Attention_Loss_001nd"

python main.py ddd --arch dla_34 --batch_size 8 --val_intervals 2 --master_batch 7 --num_epochs 140 --lr_step 45,60 --checkpoints_path "/root/CenterNet/checkpoints/Resnet34_Light_FPN_Attention_Loss_002nd"
python evaluate.py ddd --arch dla_34 --checkpoints_path "/root/CenterNet/checkpoints/Resnet34_Light_FPN_Attention_Loss_002nd"

python main.py ddd --arch dla_34 --batch_size 8 --val_intervals 2 --master_batch 7 --num_epochs 140 --lr_step 45,60 --checkpoints_path "/root/CenterNet/checkpoints/Resnet34_Light_FPN_Attention_Loss_003nd"
python evaluate.py ddd --arch dla_34 --checkpoints_path "/root/CenterNet/checkpoints/Resnet34_Light_FPN_Attention_Loss_003nd"

python main.py ddd --arch dla_34 --batch_size 8 --val_intervals 2 --master_batch 7 --num_epochs 140 --lr_step 45,60 --checkpoints_path "/root/CenterNet/checkpoints/Resnet34_Light_FPN_Attention_Loss_004nd"
python evaluate.py ddd --arch dla_34 --checkpoints_path "/root/CenterNet/checkpoints/Resnet34_Light_FPN_Attention_Loss_004nd"
