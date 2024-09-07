import torch
import torch.nn.functional as F
import numpy as np
import dataloader4kg
from kgcn import KGCN


class KG:
    def __init__(self, model_path, entity_mapping_path, device="cuda:0" if torch.cuda.is_available() else'cpu'):
        self.device = torch.device(device)
        self.model_path = model_path
        self.entity_mapping_path = entity_mapping_path
        self.model = None
        self.entity_id_to_name = {}
        self.new_set = None
        self.users = None

    def load_model(self):
        with open(self.entity_mapping_path, 'r', encoding='utf-8') as file:
            for line in file:
                name, id_str = line.strip().split('\t')
                entity_id = int(id_str)
                self.entity_id_to_name[entity_id] = name
                # 创建并加载模型
        self.users, items, train_set, test_set, self.new_set = dataloader4kg.read_ClickData(dataloader4kg.Travel_yun.RATING, dataloader4kg.Travel_yun.rating1)

        entitys, relations, kgTriples = dataloader4kg.read_KG(dataloader4kg.Travel_yun.KG)
        adj_kg = dataloader4kg.construct_kg(kgTriples)
        adj_entity, adj_relation = dataloader4kg.construct_adj(10, adj_kg, len(entitys))
        # 将邻接矩阵转移到GPU上
        adj_entity = torch.LongTensor(adj_entity).to(self.device)
        adj_relation = torch.LongTensor(adj_relation).to(self.device)
        self.model = KGCN(max(self.users) + 1, n_entitys=max(entitys) + 1, n_relations=max(relations) + 1,
                     e_dim=10, adj_entity=adj_entity, adj_relation=adj_relation,
                     n_neighbors=10,
                     aggregator_method='sum',
                     act_method=F.relu,
                     drop_rate=0.5).to(self.device)
        return self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device(self.device)))

    def recommend_for_user(self, top_k=10):
        new_set = torch.LongTensor(self.new_set)
        self.model.eval()
        with torch.no_grad():
            user_ids = new_set[:, 0].to(self.device)
            item_ids = new_set[:, 1].to(self.device)
            logits = self.model(user_ids, item_ids, True)
            predictions = [1 if i >= 0.5 else 0 for i in logits]

            top_k_items = np.argsort(predictions)[::-1][:top_k]

            recommended_entity_names = [self.entity_id_to_name[entity_id] for entity_id in top_k_items]

            return  recommended_entity_names


if __name__ == '__main__':
    # 设置推荐景点数
    n_neighbors = 10
    model_path = './my-kgcn/model/model.pth'
    entity_mapping_path = './my-kgcn/data/encoded_entities.tsv'

    recommendation_model = KG(model_path, entity_mapping_path)

    # 加载模型
    recommendation_model.load_model()

    top_k_recommendations = recommendation_model.recommend_for_user(top_k=n_neighbors)
    print(top_k_recommendations)
