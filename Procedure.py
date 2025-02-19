import numpy as np
import torch
import utils
import world


def BPR_train_original(dataset, recommend_model, loss_class, epoch):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    allusers = list(range(dataset.n_users))
    S, sam_time = utils.UniformSample_original(allusers, dataset)
    # print(f"BPR[sample time][{sam_time[0]:.1f}={sam_time[1]:.2f}+{sam_time[2]:.2f}]")
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg, epoch)
        aver_loss += cri
    # cl_loss = bpr.stageTwo()
    aver_loss = aver_loss / total_batch
    return aver_loss


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}


def Test(dataset, Recmodel, epoch, cold=False, w=None):
    u_batch_size = world.config['test_u_batch_size']
    if cold:
        testDict: dict = dataset.coldTestDict   # cold start users
    else:
        testDict: dict = dataset.testDict
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)


            rating = Recmodel.getUsersRating(batch_users_gpu)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))


        if world.dataset == 'lastfm':
            if cold:
                print(f"Precision@10: {results['precision'][0]:.4f}  {((results['precision'][0] - 0.0458)/0.0458)*100:.2f}%")
                print(f"Precision@20: {results['precision'][1]:.4f}  {((results['precision'][1] - 0.0333)/0.0333)*100:.2f}%")
                # print(f"Precision@50: {results['precision'][2]:.4f}")
                print(f"Recall@10   : {results['recall'][0]:.4f}  {((results['recall'][0] - 0.1974)/0.1974)*100:.2f}%")
                print(f"Recall@20   : {results['recall'][1]:.4f}  {((results['recall'][1] - 0.2663)/0.2663)*100:.2f}%")
                # print(f"Recall@50   : {results['recall'][2]:.4f} ")
                print(f"NDCG@10     : {results['ndcg'][0]:.4f}  {((results['ndcg'][0] - 0.1419)/0.1419)*100:.2f}%")
                print(f"NDCG@20     : {results['ndcg'][1]:.4f}  {((results['ndcg'][1] - 0.1643)/0.1643)*100:.2f}%")
                # print(f"NDCG@50     : {results['ndcg'][2]:.4f} ")
            else:
                print(f"Precision@10: {results['precision'][0]:.4f}  {((results['precision'][0] - 0.1972)/0.1972)*100:.2f}%")
                print(f"Precision@20: {results['precision'][1]:.4f}  {((results['precision'][1] - 0.1368)/0.1368)*100:.2f}%")
                # print(f"Precision@50: {results['precision'][2]:.4f}")
                print(f"Recall@10   : {results['recall'][0]:.4f}  {((results['recall'][0] - 0.2026)/0.2026)*100:.2f}%")
                print(f"Recall@20   : {results['recall'][1]:.4f}  {((results['recall'][1] - 0.2794)/0.2794)*100:.2f}%")
                # print(f"Recall@50   : {results['recall'][2]:.4f} ")
                print(f"NDCG@10     : {results['ndcg'][0]:.4f}  {((results['ndcg'][0] - 0.2566)/0.2566)*100:.2f}%")
                print(f"NDCG@20     : {results['ndcg'][1]:.4f}  {((results['ndcg'][1] - 0.2883)/0.2883)*100:.2f}%")
                # print(f"NDCG@50     : {results['ndcg'][2]:.4f} ")

        elif world.dataset == 'ciao':
            if cold:
                print(f"Precision@10: {results['precision'][0]:.4f}  {((results['precision'][0] - 0.0134)/0.0134)*100:.2f}%")
                print(f"Precision@20: {results['precision'][1]:.4f}  {((results['precision'][1] - 0.0097)/0.0097)*100:.2f}%")
                # print(f"Precision@50: {results['precision'][2]:.4f}")
                print(f"Recall@10   : {results['recall'][0]:.4f}  {((results['recall'][0] - 0.0441)/0.0441)*100:.2f}%")
                print(f"Recall@20   : {results['recall'][1]:.4f}  {((results['recall'][1] - 0.0630)/0.0630)*100:.2f}%")
                # print(f"Recall@50   : {results['recall'][2]:.4f} ")
                print(f"NDCG@10     : {results['ndcg'][0]:.4f}  {((results['ndcg'][0] - 0.0328)/0.0328)*100:.2f}%")
                print(f"NDCG@20     : {results['ndcg'][1]:.4f}  {((results['ndcg'][1] - 0.0394)/0.0394)*100:.2f}%")
                # print(f"NDCG@50     : {results['ndcg'][2]:.4f} ")
            else:
                print(f"Precision@10: {results['precision'][0]:.4f}  {((results['precision'][0] - 0.0276)/0.0276)*100:.2f}%")
                print(f"Precision@20: {results['precision'][1]:.4f}  {((results['precision'][1] - 0.0205)/0.0205)*100:.2f}%")
                # print(f"Precision@50: {results['precision'][2]:.4f}")
                print(f"Recall@10   : {results['recall'][0]:.4f}  {((results['recall'][0] - 0.0430)/0.0430)*100:.2f}%")
                print(f"Recall@20   : {results['recall'][1]:.4f}  {((results['recall'][1] - 0.0618)/0.0618)*100:.2f}%")
                # print(f"Recall@50   : {results['recall'][2]:.4f} ")
                print(f"NDCG@10     : {results['ndcg'][0]:.4f}  {((results['ndcg'][0] - 0.0441)/0.0441)*100:.2f}%")
                print(f"NDCG@20     : {results['ndcg'][1]:.4f}  {((results['ndcg'][1] - 0.0486)/0.0486)*100:.2f}%")
                # print(f"NDCG@50     : {results['ndcg'][2]:.4f} ")
        return results
