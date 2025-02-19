import world
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import linalg as LA
import mmids
from tqdm import tqdm

class PureBPR(nn.Module):
    def __init__(self, config, dataset):
        super(PureBPR, self).__init__()
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        print("using Normal distribution initializer")

    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb = self.embedding_item(pos.long())
        neg_emb = self.embedding_item(neg.long())
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        return loss, reg_loss

class LightGCN(nn.Module):
    def __init__(self, config, dataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self._init_weight()

    def _init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['layer']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.f = nn.Sigmoid()

        self.interactionGraph = self.dataset.getInteractionGraph()
        # self.reconstructGraph = self.dataset.getReconstructGraph()

        print(f"{world.model_name} is already to go")

    def computer(self):
        """
        propagate methods for lightGCN
        1. using lightGCN get the embeddings of all users and items
        2. using SVD reconstruct the social graph and get the embeddings of all users
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        # torch.split(all_emb , [self.num_users, self.num_items])
        
        embs = [all_emb]
        G = self.interactionGraph
        # G_recon = self.reconstructGraph  # reconstruct graph by SVD

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(G, all_emb)

            # recon_user = torch.sparse.mm(G_recon, users_emb)
            embs.append(all_emb)
            # recon_user.append(recon_user)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        self.final_user, self.final_item = users, items
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.final_user, self.final_item
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego


    def bpr_loss(self, users, pos, neg):

        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long()) # 用 lightGCN 得到的 user, pos, neg embeddings
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss


    #############  LightGCL CL loss ############# 
    # G_u_norm = self.G_u  # user enbedding from SVD propagation
    # E_u_norm = self.E_u  # user embedding from GNN propagation
    # G_i_norm = self.G_i  # item embedding from SVD propagation
    # E_i_norm = self.E_i  # item embedding from GNN propagation

    # neg_score = torch.log(torch.exp(G_u_norm[uids] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
    # neg_score += torch.log(torch.exp(G_i_norm[iids] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
    # pos_score = (torch.clamp((G_u_norm[uids] * E_u_norm[uids]).sum(1) / self.temp,-5.0,5.0)).mean() + (torch.clamp((G_i_norm[iids] * E_i_norm[iids]).sum(1) / self.temp,-5.0,5.0)).mean()
    # loss_s = -pos_score + neg_score
    #############  LightGCL CL loss ############# 


class CSRec(LightGCN):
    def _init_weight(self):
        super(CSRec, self)._init_weight()
        self.socialGraph = self.dataset.getSocialGraph()
        self.Graph_Comb = Graph_Comb(self.latent_dim)
        # perform svd reconstruction
        print('Performing SVD...')
        print(self.interactionGraph.size())

        svd_u,s,_ = self.svd(self.interactionGraph, l=3)  # l is the rank
        svd_user,_ = torch.split(svd_u, [self.num_users, self.num_items])
        self.reconSocialGraph = svd_user @ (torch.diag(s)) @ svd_user.T
        del s
        print('SVD done.')
        self.cl_users_list = self.cl_users(self.config['percentage'], False)
        # self.reconstructGraph = self.dataset.getReconstructGraph()  


    def svd(self, A, l, maxiter=100):
        # seed = 535
        # rng = np.random.default_rng(seed)
        # rng.uniform()
        # V = rng.normal(0,1,(np.size(A.numpy(),1),l))
        V = torch.randn(A.shape[1], l).to(A.device)
        for _ in tqdm(range(maxiter)):
            W = torch.sparse.mm( A, V )
            Z = torch.sparse.mm( A.t(), W )
            V, R = mmids.gramschmidt(Z)
        W = torch.sparse.mm(A, V)
        S = torch.norm(W,dim=0)
        U = torch.stack([W[:,i]/S[i] for i in range(W.shape[1])],dim=-1)
        return U, S, V


    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        
        A = self.interactionGraph
        S = self.socialGraph
        R = self.reconSocialGraph

        embs = [all_emb]
        embs_recon = [users_emb]
        users_emb_recon = users_emb

        for layer in range(self.n_layers):
            # embedding from last layer
            users_emb, items_emb = torch.split(all_emb, [self.num_users, self.num_items])
            # social network propagation(user embedding)

            users_emb_social = torch.sparse.mm(S, users_emb)
            # user-item bi-network propagation(user and item embedding)
            all_emb_interaction = torch.sparse.mm(A, all_emb)
            
            users_emb_recon = torch.matmul(R, users_emb_recon)


            # get users_emb_interaction
            users_emb_interaction, items_emb_next = torch.split(all_emb_interaction, [self.num_users, self.num_items])


            # graph fusion model
            users_emb_next = self.Graph_Comb(users_emb_social, users_emb_interaction)
            all_emb = torch.cat([users_emb_next, items_emb_next])
            

            embs.append(all_emb)
            embs_recon.append(users_emb_recon)


        embs = torch.stack(embs, dim=1)
        embs_recon = torch.stack(embs_recon, dim=1)

        final_embs = torch.mean(embs, dim=1)
        final_embs_recon = torch.mean(embs_recon, dim=1)    
        
        self.user_recon = final_embs_recon

        users, items = torch.split(final_embs, [self.num_users, self.num_items])
        self.final_user, self.final_item = users, items
        return users, items


    def cl_users(self, percentage, social_mask):
        A = self.interactionGraph
        S = self.socialGraph
        num_users = S.shape[0]

        items_count = []
        for i in range(num_users):
            items_count.append(A[i].coalesce().indices().shape[1])

        sorted_items_count = sorted(items_count)
        index = int(len(sorted_items_count) * percentage)
        threshold = sorted_items_count[index]
        u2i_mask = torch.tensor(items_count) < threshold   
        u2i_mask = u2i_mask.to(S.device)

        if social_mask:
            # S = self.socialGraph
            # num_users = S.shape[0]

            idx = S.coalesce().indices()
            val = torch.ones(idx.shape[1]).to(S.device)
            count_graph = torch.sparse_coo_tensor(idx, val, torch.Size([num_users, num_users]))
            social_count = torch.sum(count_graph.coalesce().to_dense(),dim=1)
            avg_social = sum(social_count) / len(social_count)
            u2u_mask = social_count > avg_social 

            users_mask = torch.logical_and(u2u_mask, u2i_mask)  # selecting user who have many social interactions but less item interactions
            cl_users_list = users_mask.nonzero().squeeze().tolist()
        # print(f"Selected users: {len(selected_users)}, {len(selected_users) / num_users * 100:.2f}%")
        else:
            cl_users_list = u2i_mask.nonzero().squeeze().tolist()
        
        # cl_user = self.final_user[users_mask]
        # cl_user_recon = self.user_recon[users_mask]
        # cl_users = torch.nonzero(users_mask).squeeze()
        return cl_users_list


    def cl_loss(self, users):
        # Self-supervised Graph Learning for Recommendation
        # contrastive loss
        
        # self.cl_users_list
        # users.tolist()
        intersection = list(set(self.cl_users_list) & set(users.tolist()))

        cl_user = self.final_user[intersection]
        cl_user_recon = self.user_recon[intersection]

        # print(f"the # selcted users: {len(cl_user)}")

        cl_user_norm = torch.norm(cl_user, dim=1, keepdim=True)
        cl_user = cl_user / (cl_user_norm + 1e-8 )
        cl_user_recon_norm = torch.norm(cl_user_recon, dim=1, keepdim=True)
        cl_user_recon = cl_user_recon / (cl_user_recon_norm + 1e-8)
        
        gram = torch.matmul(cl_user, cl_user_recon.T)
        diag = torch.diag(gram)
        
        # 剔除对角线元素
        mask = torch.eye(len(diag), dtype=torch.bool)
        if gram.numel() != 0:
            masked_gram = gram[~mask].reshape(len(diag), -1)
            pos_score = torch.log(torch.exp(diag / 0.2) + 1e-8).sum()               # cl loss temperature = 0.2
            neg_score = torch.log(torch.exp(masked_gram / 0.2).sum(1) + 1e-8).sum() # cl loss temperature = 0.2
        else:
            pos_score = 0
            neg_score = 0
        loss = - pos_score + neg_score
        return loss


class Graph_Comb(nn.Module):
    def __init__(self, embed_dim):
        super(Graph_Comb, self).__init__()
        self.att_x = nn.Linear(embed_dim, embed_dim, bias=False)
        self.att_y = nn.Linear(embed_dim, embed_dim, bias=False)
        self.comb = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x, y):
        h1 = torch.tanh(self.att_x(x))
        h2 = torch.tanh(self.att_y(y))
        output = self.comb(torch.cat((h1, h2), dim=1))
        output = output / output.norm(2)
        return output