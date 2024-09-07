import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm #产生进度条
import dataloader4kg
from sklearn.metrics import precision_score,recall_score,accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AttentionLayer(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionLayer, self).__init__()
        self.fc = nn.Linear(embed_dim * 3, 1)

    def forward(self, entity_embed, neighbor_entity_embed, relation_embed):
        # Concatenate embeddings
        combined = torch.cat([entity_embed, neighbor_entity_embed, relation_embed], dim=1)
        # Apply the fully connected layer
        attn_scores = self.fc(combined)
        attn_scores = F.softmax(attn_scores, dim=1)
        return attn_scores


class KGCN( nn.Module ):

    def __init__( self, n_users, n_entitys, n_relations,
                  e_dim,  adj_entity, adj_relation, n_neighbors,
                  aggregator_method = 'sum',
                  act_method = F.relu, drop_rate=0.5):
        super( KGCN, self ).__init__()

        self.e_dim = e_dim  # 特征向量维度
        self.aggregator_method = aggregator_method #消息聚合方法
        self.n_neighbors = n_neighbors #邻居的数量
        self.user_embedding = nn.Embedding( n_users, e_dim, max_norm = 1 )
        self.entity_embedding = nn.Embedding( n_entitys, e_dim, max_norm = 1)
        self.relation_embedding = nn.Embedding( n_relations, e_dim, max_norm = 1)

        # self.attention_layer = AttentionLayer(e_dim)
        self.adj_entity = adj_entity #节点的邻接列表
        self.adj_relation = adj_relation #关系的邻接列表

        #线性层
        self.linear_layer = nn.Linear(
                in_features = self.e_dim * 2 if self.aggregator_method == 'concat' else self.e_dim,
                out_features = self.e_dim,
                bias = True)

        self.act = act_method #激活函数
        self.drop_rate = drop_rate #drop out 的比率

    def forward(self, users, items, is_evaluate = False):
        # 将数据移到GPU上
        users = users.to(device)
        items = items.to(device)
        neighbor_entitys, neighbor_relations = self.get_neighbors( items )
        user_embeddings = self.user_embedding( users)
        item_embeddings = self.entity_embedding( items )

        #得到v波浪线
        neighbor_vectors = self.__get_neighbor_vectors( neighbor_entitys, neighbor_relations, user_embeddings )

        out_item_embeddings = self.aggregator( item_embeddings, neighbor_vectors,is_evaluate)

        out = torch.sigmoid( torch.sum( user_embeddings * out_item_embeddings, axis = -1 ) )

        return out

    def get_neighbors( self, items ):#得到邻居的节点embedding,和关系embedding
        #[[1,2,3,4,5],[2,1,3,4,5]...[]]#总共batchsize个n_neigbor的id
        # 将items转移到GPU上
        items = items.to(device)
        entity_ids = [ self.adj_entity[item] for item in items ]
        relation_ids = [ self.adj_relation[item] for item in items ]
        # 使用torch.tensor代替torch.LongTensor，并直接在GPU上创建张量
        neighbor_entities = [torch.unsqueeze(self.entity_embedding(one_ids.clone().detach()), 0) for one_ids in
                             entity_ids]
        neighbor_relations = [torch.unsqueeze(self.relation_embedding(one_ids.clone().detach()), 0) for one_ids in
                              relation_ids]
        # [batch_size, n_neighbor, dim]
        neighbor_entities = torch.cat( neighbor_entities, dim=0 )
        neighbor_relations = torch.cat( neighbor_relations, dim=0 )

        return neighbor_entities, neighbor_relations

    #得到v波浪线
    def __get_neighbor_vectors(self, neighbor_entitys, neighbor_relations, user_embeddings):
        # [batch_size, n_neighbor, dim]
        user_embeddings = torch.cat([torch.unsqueeze(user_embeddings,1) for _ in range(self.n_neighbors)],dim=1)
        # [batch_size, n_neighbor]
        user_relation_scores = torch.sum(user_embeddings * neighbor_relations, axis=2)
        # [batch_size, n_neighbor]
        user_relation_scores_normalized = F.softmax(user_relation_scores, dim=-1)
        # [batch_size, n_neighbor, 1]
        user_relation_scores_normalized = torch.unsqueeze(user_relation_scores_normalized, 2)
        # [batch_size, dim]
        neighbor_vectors = torch.sum(user_relation_scores_normalized * neighbor_entitys, axis=1)
        return neighbor_vectors

    #经过进一步的聚合与线性层得到v
    def aggregator(self,item_embeddings, neighbor_vectors, is_evaluate):
        # [batch_size, dim]
        if self.aggregator_method == 'sum':
            output = item_embeddings + neighbor_vectors
        elif self.aggregator_method == 'concat':
            # [batch_size, dim * 2]
            output = torch.cat([item_embeddings, neighbor_vectors], axis=-1)
        else:#neighbor
            output = neighbor_vectors

        if not is_evaluate:
            output = F.dropout(output, self.drop_rate)
        # [batch_size, dim]
        output = self.linear_layer(output)
        return self.act(output)

#验证
def do_evaluate( model, testSet ):
    testSet = torch.LongTensor(testSet)
    model.eval()
    with torch.no_grad():
        user_ids = testSet[:, 0]
        item_ids = testSet[:, 1]
        labels = testSet[:, 2]
        logits = model( user_ids, item_ids, True )
        predictions = [1 if i >= 0.5 else 0 for i in logits]
        p = precision_score(y_true = labels, y_pred = predictions)
        r = recall_score(y_true = labels, y_pred = predictions)
        acc = accuracy_score(labels, y_pred = predictions)
        return p,r,acc

def do_predict(model, newSet, top_k=10):
    newSet = torch.LongTensor(newSet)
    model.eval()
    with torch.no_grad():
        user_ids = newSet[:, 0]
        item_ids = newSet[:, 1]
        labels = newSet[:, 2]
        logits = model(user_ids, item_ids, True)
        predictions = [1 if i >= 0.5 else 0 for i in logits]
        p = precision_score(y_true=labels, y_pred=predictions)
        r = recall_score(y_true=labels, y_pred=predictions)
        acc = accuracy_score(labels, y_pred=predictions)
        top_k_items = np.argsort(predictions)[::-1][:top_k]

        return p, r, acc, top_k_items




def train( epochs, batchSize, lr,
           n_users, n_entitys, n_relations,
           adj_entity, adj_relation,
           train_set, test_set,
           n_neighbors,
           aggregator_method = 'sum',
           act_method = F.relu, drop_rate = 0.5, weight_decay=5e-4
         ):

    # 记录损失和评估指标
    losses = []
    precisions = []
    recalls = []
    accuracies = []

    model = KGCN( n_users, n_entitys, n_relations,
                  10, adj_entity, adj_relation,
                  n_neighbors = n_neighbors,
                  aggregator_method = aggregator_method,
                  act_method = act_method,
                  drop_rate = drop_rate ).to(device)
    optimizer = torch.optim.Adam( model.parameters(), lr = lr, weight_decay = weight_decay )
    loss_fcn = nn.BCELoss()
    dataIter = dataloader4kg.DataIter()

    for epoch in range( epochs ):
        total_loss = 0.0
        for datas in tqdm( dataIter.iter( train_set, batchSize = batchSize ) ):
            # 将数据移到GPU上
            datas = datas.to(device)
            user_ids = datas[:, 0]
            item_ids = datas[:, 1]
            labels = datas[:, 2]
            logits = model.forward( user_ids, item_ids )
            loss = loss_fcn( logits, labels.float() )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        p, r, acc = do_evaluate(model,test_set)


        # 在每个epoch结束时记录
        losses.append(total_loss / (len(train_set) // batchSize))
        p, r, acc = do_evaluate(model, test_set)
        precisions.append(p)
        recalls.append(r)
        accuracies.append(acc)

    # 保存模型
    # 保存模型的路径
    model_path = os.path.join('model', 'model.pth')
    torch.save(model.state_dict(), model_path)
    return losses, precisions, recalls, accuracies, model


# 定义推荐函数
def recommend_new_user(model, new_user_id, all_items, top_k=10):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        user_ids = torch.tensor([new_user_id], device=device).repeat(len(all_items))
        item_ids = torch.tensor(all_items, device=device)
        predictions = model(user_ids, item_ids, is_evaluate=True).cpu().numpy()
        top_k_items = np.argsort(predictions)[::-1][:top_k]
        return top_k_items


if __name__ == '__main__':
    n_neighbors = 10

    users, items, train_set, test_set, new_set = dataloader4kg.read_ClickData(dataloader4kg.Travel_yun.RATING, dataloader4kg.Travel_yun.rating1)
    entitys, relations, kgTriples = dataloader4kg.read_KG(dataloader4kg.Travel_yun.KG)
    adj_kg = dataloader4kg.construct_kg(kgTriples)
    adj_entity, adj_relation = dataloader4kg.construct_adj(n_neighbors, adj_kg, len(entitys))
    # 将邻接矩阵转移到GPU上
    adj_entity = torch.LongTensor(adj_entity).to(device)
    adj_relation = torch.LongTensor(adj_relation).to(device)

    # 训练模型并获取结果
    losses, precisions, recalls, accuracies, model = train(epochs = 200, batchSize = 1024, lr = 0.01,
           n_users = max( users ) + 1, n_entitys = max( entitys ) + 1,
           n_relations = max( relations ) + 1, adj_entity = adj_entity,
           adj_relation = adj_relation, train_set = train_set,
           test_set = test_set, n_neighbors = n_neighbors,
           aggregator_method = 'sum', act_method = F.relu, drop_rate = 0.5 )

    # # 绘制损失图
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(losses, label='Loss')
    # plt.title('Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    #
    # # 绘制评估指标图
    # plt.subplot(1, 2, 2)
    # plt.plot(precisions, label='Precision')
    # plt.plot(recalls, label='Recall')
    # plt.plot(accuracies, label='Accuracy')
    # plt.title('Evaluation Metrics')
    # plt.xlabel('Epoch')
    # plt.ylabel('Metric Value')
    # plt.legend()
    #
    # plt.show()

    user_id = 250  # 用户的ID
    all_items = list(range(max(items) + 1))  # 假设所有物品的ID列表

    top_k_recommendations = recommend_new_user(model, user_id, all_items, top_k=10)



    entity_id_to_name = {}

    with open('C:\\Users\\ASUS\\Desktop\\AI\\KG\\kgcn\\my-kgcn\\data\\encoded_entities.tsv', 'r', encoding='utf-8') as file:
        for line in file:
            name, id_str = line.strip().split('\t')
            entity_id = int(id_str)
            entity_id_to_name[entity_id] = name


    recommended_entity_ids = top_k_recommendations

    # 将推荐的实体ID转换为名称
    recommended_entity_names = [entity_id_to_name[entity_id] for entity_id in recommended_entity_ids]



    p, r, acc= do_evaluate(model, test_set)

    p, r, acc, top_k = do_predict(model, new_set)

    names = [entity_id_to_name[entity_id] for entity_id in top_k]

