for pred_len in 24 36 48 60
do
    python -u run.py \
      --random_seed 2021 \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path national_illness.csv \
      --model_id national_illness \
      --model DGCformer \
      --data custom \
      --features M \
      --seq_len 104 \
      --pred_len $pred_len \
      --enc_in 7 \
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
      --train_epochs 200\
      --lradj 'constant'\
      --itr 1 --batch_size 16 --learning_rate 0.0025 \
      --patience 100
done