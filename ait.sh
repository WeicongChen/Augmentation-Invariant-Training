#!bin/sh

gpu=2
lr=0.1
batch_size=128
test_batch_size=256
date=$(date +%m%d%H%M)
algo=0
if [ $algo -eq 0 ] ; then # for baseline
    data="cifar100"
    arch="resnet32_rp_"${data}
    kl_w=0.
    parallel=1
elif [ $algo -eq 1 ] ; then # for AIT
    data="cifar100"
    arch="mobilev2_rp,rp_"${data}
    arch="wideresnet_rp,rp_"${data}
    arch="densenet_rp,rp_"${data}
    arch="resnet32_rp,rp_"${data}
    kl_w=1.
    parallel=2
elif [ $algo -eq 2 ] ; then # for multi-AIT
    arch="mobilev2,resnet32_rp,rp_cifar100"
    arch="resnet32,resnet32_rp,rp_cifar100"
    kl_w=1.
    parallel=1
fi
algo=${arch}"_"${kl_w}"_"${parallel}"_"${date}
log="logs/log-"${algo}".log"

CUDA_VISIBLE_DEVICES=$gpu python ait.py \
    --batch-size $batch_size \
    --test-batch-size $test_batch_size \
    --arch $arch \
    --parallel $parallel \
    --lr $lr \
    --algo $algo \
    --kl-w $kl_w \
    --log $log
