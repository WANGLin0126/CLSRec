import time
from os.path import join
import torch
import Procedure
import register
import utils
import world
from register import dataset

# ==============================

utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
torch.autograd.set_detect_anomaly(True)
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)
# 
weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        print(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")

best_ndcg, best_recall, best_pre = 0, 0, 0
best_ndcg_20, best_recall_20, best_pre_20 = 0, 0, 0
best_ndcg_cold, best_recall_cold, best_pre_cold = 0, 0, 0
low_count, low_count_cold = 0, 0
try:
    for epoch in range(world.TRAIN_epochs + 1):
        # print('======================')
        print(f'EPOCH[{epoch}/{world.TRAIN_epochs}]', end=' ')
        start = time.time()
        if epoch % 50 == 1 or epoch == world.TRAIN_epochs:
            print("[TEST]")
            results = Procedure.Test(dataset, Recmodel, epoch, False)
            results_cold = Procedure.Test(dataset, Recmodel, epoch, True)
            if results['ndcg'][0] < best_ndcg:
                low_count += 1
                if low_count == 30:
                    if epoch > 1000:
                        break
                    else:
                        low_count = 0
            else:
                best_recall = results['recall'][0]
                best_ndcg   = results['ndcg'][0]
                best_pre    = results['precision'][0]
                # low_count   = 0

            # if results['ndcg'][1] < best_ndcg_20:
            #     low_count += 1
            #     if low_count == 30:
            #         if epoch > 1000:
            #             break
            #         else:
            #             low_count = 0
            # else:
                best_recall_20 = results['recall'][1]
                best_ndcg_20   = results['ndcg'][1]
                best_pre_20    = results['precision'][1]
                low_count   = 0

            if results_cold['ndcg'][0] > best_ndcg_cold:

                best_recall_cold = results_cold['recall'][0]
                best_ndcg_cold = results_cold['ndcg'][0]
                best_pre_cold = results_cold['precision'][0]

                best_recall_cold_20 = results_cold['recall'][1]
                best_ndcg_cold_20 = results_cold['ndcg'][1]
                best_pre_cold_20 = results_cold['precision'][1]

        loss = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch)
        print(f'[saved][BPR aver loss {loss:.4e}]')
        torch.save(Recmodel.state_dict(), weight_file)

finally:
    if world.dataset == 'lastfm':
        print(f"Best Precision@10   :{best_pre:.4f}  {((best_pre - 0.1972)/0.1972)*100:.2f}%")
        print(f"Best Precision@20   :{best_pre_20:.4f}  {((best_pre_20 - 0.1368)/0.1368)*100:.2f}%")
        print(f"Best Recall@10      :{best_recall:.4f}  {((best_recall - 0.2026)/0.2026)*100:.2f}%")
        print(f"Best Recall@20      :{best_recall_20:.4f}  {((best_recall_20 - 0.2794)/0.2794)*100:.2f}%")
        print(f"Best NDCG@10        :{best_ndcg:.4f}  {((best_ndcg - 0.2566)/0.2566)*100:.2f}%")
        print(f"Best NDCG@20        :{best_ndcg_20:.4f}  {((best_ndcg_20 - 0.2883)/0.2883)*100:.2f}%")   

        print(f"--------------Cold Start-----------------")
        print(f"Best Precision@10   :{best_pre_cold:.4f}  {((best_pre_cold - 0.0458)/0.0458)*100:.2f}%")
        print(f"Best Precision@20   :{best_pre_cold_20:.4f}  {((best_pre_cold_20 - 0.0333)/0.0333)*100:.2f}%")
        print(f"Best Recall@10      :{best_recall_cold:.4f}  {((best_recall_cold - 0.1974)/0.1974)*100:.2f}%")
        print(f"Best Recall@20      :{best_recall_cold_20:.4f}  {((best_recall_cold_20 - 0.2663)/0.2663)*100:.2f}%")
        print(f"Best NDCG@10        :{best_ndcg_cold:.4f}  {((best_ndcg_cold - 0.1419)/0.1419)*100:.2f}%")
        print(f"Best NDCG@20        :{best_ndcg_cold_20:.4f}  {((best_ndcg_cold_20 - 0.1643)/0.1643)*100:.2f}%")
    
    elif world.dataset == 'ciao':
        print(f"Precision@10        :{best_pre:.4f}  {((best_pre - 0.0276)/0.0276)*100:.2f}%")
        print(f"Precision@20        :{best_pre_20:.4f}  {((best_pre_20 - 0.0205)/0.0205)*100:.2f}%")
        print(f"Recall@10           :{best_recall:.4f}  {((best_recall - 0.0430)/0.0430)*100:.2f}%")
        print(f"Recall@20           :{best_recall_20:.4f}  {((best_recall_20 - 0.0618)/0.0618)*100:.2f}%")
        print(f"NDCG@10             :{best_ndcg:.4f}  {((best_ndcg - 0.0441)/0.0441)*100:.2f}%")
        print(f"NDCG@20             :{best_ndcg_20:.4f}  {((best_ndcg_20 - 0.0486)/0.0486)*100:.2f}%")

        print(f"--------------Cold Start-----------------")
        print(f"Precision@10        :{best_pre_cold:.4f}  {((best_pre_cold - 0.0134)/0.0134)*100:.2f}%")
        print(f"Precision@20        :{best_pre_cold_20:.4f}  {((best_pre_cold_20 - 0.0097)/0.0097)*100:.2f}%")
        print(f"Recall@10           :{best_recall_cold:.4f}  {((best_recall_cold  - 0.0441)/0.0441)*100:.2f}%")
        print(f"Recall@20           :{best_recall_cold_20:.4f}  {((best_recall_cold_20 - 0.0630)/0.0630)*100:.2f}%")
        print(f"NDCG@10             :{best_ndcg_cold:.4f}  {((best_ndcg_cold - 0.0328)/0.0328)*100:.2f}%")
        print(f"NDCG@20             :{best_ndcg_cold_20:.4f}  {((best_ndcg_cold_20 - 0.0394)/0.0394)*100:.2f}%")

