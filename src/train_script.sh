python main.py ddd --arch res_18 --batch_size 8 --val_intervals 2 --master_batch 7 --num_epochs 70 --lr_step 45,60 --checkpoints_path "/root/CenterNet/checkpoints/Light_FPN_Attention_Loss_Plus_Rot_001nd"
python evaluate.py ddd --arch res_18 --checkpoints_path "/root/CenterNet/checkpoints/Light_FPN_Attention_Loss_Plus_Rot_001nd"

python main.py ddd --arch res_18 --batch_size 8 --val_intervals 2 --master_batch 7 --num_epochs 70 --lr_step 45,60 --checkpoints_path "/root/CenterNet/checkpoints/Light_FPN_Attention_Loss_Plus_Rot_002nd"
python evaluate.py ddd --arch res_18 --checkpoints_path "/root/CenterNet/checkpoints/Light_FPN_Attention_Loss_Plus_Rot_002nd"

python main.py ddd --arch res_18 --batch_size 8 --val_intervals 2 --master_batch 7 --num_epochs 70 --lr_step 45,60 --checkpoints_path "/root/CenterNet/checkpoints/Light_FPN_Attention_Loss_Plus_Rot_003nd"
python evaluate.py ddd --arch res_18 --checkpoints_path "/root/CenterNet/checkpoints/Light_FPN_Attention_Loss_Plus_Rot_003nd"

python main.py ddd --arch res_18 --batch_size 8 --val_intervals 2 --master_batch 7 --num_epochs 70 --lr_step 45,60 --checkpoints_path "/root/CenterNet/checkpoints/Light_FPN_Attention_Loss_Plus_Rot_004nd"
python evaluate.py ddd --arch res_18 --checkpoints_path "/root/CenterNet/checkpoints/Light_FPN_Attention_Loss_Plus_Rot_004nd"