import torch
import torch.nn as nn
from model.News_embedding import News_embedding
from model.core import CapsuleLayer

class User_modeling(nn.Module):

    def __init__(self, config, user_history_dict, news_embedding_dim, user_embedding_dim, doc_feature_dict, entity_embedding, relation_embedding, adj_entity, adj_relation, entity_num, position_num, type_num):
        super(User_modeling, self).__init__()
        self.config = config
        self.user_history_dict = user_history_dict
        self.news_embedding_dim = news_embedding_dim
        self.user_embedding_dim = user_embedding_dim
        self.doc_feature_dict = doc_feature_dict
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation
        self.news_embedding = News_embedding(config, doc_feature_dict, entity_embedding, relation_embedding, adj_entity, adj_relation, entity_num, position_num, type_num)

        self.user_attention_layer1 = nn.Linear(news_embedding_dim, self.config['model']['layer_dim'])
        self.user_attention_layer2 = nn.Linear(self.config['model']['layer_dim'], 1)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim = 0)
        self.capsule_layer = CapsuleLayer(news_embedding_dim, 1, 
                    4, news_embedding_dim, 20)

    def get_user_history(self, user_id):
        user_history = []
        for userid in user_id:
            user_history.append(self.user_history_dict[userid])
        return user_history

    def user_attention_modeling(self, news_embeddings):
        print("news_embeddings:", news_embeddings.shape)
        user_attention = self.relu(self.user_attention_layer1(news_embeddings))
        user_attention = self.relu(self.user_attention_layer2(user_attention))
        user_attention_softmax = self.softmax(user_attention)
        news_attention_embedding = news_embeddings * user_attention_softmax
        user_attention_embedding = torch.sum(news_attention_embedding, dim=1)

        print("user_attention_embedding:", user_attention_embedding.shape)
        return user_attention_embedding

    def multi_interest_user_attention_modeling(self, news_embeddings):
        capsule_output = self.capsule_layer(news_embeddings)
        user_attention = self.relu(self.user_attention_layer1(news_embeddings))
        user_attention = self.relu(self.user_attention_layer2(user_attention))
        user_attention_softmax = self.softmax(user_attention)
        news_attention_embedding = news_embeddings * user_attention_softmax
        user_attention_embedding = torch.sum(news_attention_embedding, dim=1)
        multi_emb = torch.sum(capsule_output, dim=1)
        final_emb = torch.sum([user_attention_embedding, multi_emb], dim=1)
        return final_emb


    def forward(self, user_id):

        user_history = self.get_user_history(user_id)
        user_history_embedding, top_indexs = self.news_embedding(user_history)
        user_attention_modeling = self.multi_interest_user_attention_modeling(user_history_embedding)
        user_embedding = user_attention_modeling
        return user_embedding
