for pred_len in 96 192 336 720
do
    python -u run.py \
      --random_seed 2021 \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ECL.csv \
      --model_id Electricity \
      --model DGCformer \
      --data custom \
      --features M \
      --seq_len 96 \
      --pred_len 720 \
      --enc_in 321 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 64 \
      --d_ff 128 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 40\
      --lradj 'TST'\
      --pct_start 0.2\
      --itr 1 --batch_size 32 --learning_rate 0.0001
done