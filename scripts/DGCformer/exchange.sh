for pred_len in 96 192 336 720
do
    python -u run.py \
      --random_seed 2021 \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path exchange_rate.csv \
      --model_id exchange_rate \
      --model DGCformer \
      --data custom \
      --features M \
      --seq_len 336 \
      --pred_len 192 \
      --enc_in 8 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 24\
      --stride 2\
      --des 'Exp' \
      --train_epochs 100\
      --lradj 'type3'\
      --itr 1 --batch_size 24 --learning_rate 0.0001 \
      --patience 5
done