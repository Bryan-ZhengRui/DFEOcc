
#### Train model
```shell
# single gpu
python tools/train.py $config
# multiple gpu
./tools/dist_train.sh $config num_gpu
# for example
./tools/dist_train.sh projects/configs/occ_mamba_fo/occ_mamba_fo_r50.py 4
```

#### Test model
```shell
# single gpu
python tools/test.py $config $checkpoint --eval mAP
# multiple gpu
./tools/dist_test.sh $config $checkpoint num_gpu --eval mAP
```
