for pred_len in 96 192 336 720
do
    python -u run.py \
      --random_seed 2021 \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTm1.csv \
      --model_id ETTm1 \
      --model DGCformer \
      --data ETTm1 \
      --features M \
      --seq_len 96 \
      --pred_len 192 \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 256 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --lradj 'TST'\
      --pct_start 0.4\
      --itr 1 --batch_size 128 --learning_rate 0.0001 
done