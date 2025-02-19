# python main.py --dataset lastfm --cl_weight 0    --bpr_batch 2048 --lr 1e-3   --percentage 0.1 > output.txt
# python main.py --dataset lastfm --cl_weight 1e-5 --bpr_batch 4096 --lr 1e-2   --percentage 0.1 > output1.txt
# python main.py --dataset lastfm --cl_weight 1e-6 --bpr_batch 4096 --lr 1e-2   --percentage 0.1 > output2.txt

for dataset in lastfm ciao; do
    for cl_weight in 1e-5 1e-6 1e-7 1e-8; do # 1e-6 1e-8 1e-9 0
        for percentage in 0.01 0.02 0.03 0.06 0.1; do
            python main.py --bpr_batch 2048 --epochs 10000 --dataset $dataset --cl_weight $cl_weight --lr 1e-3 --percentage $percentage > outputs2/${dataset}_cl_weight_${cl_weight}_perc_${percentage}.txt
        done
    done
done