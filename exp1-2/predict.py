import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text  import TfidfVectorizer

#数据预处理
data = pd.read_csv('D:\\2024秋\\web\\webinfoexp\\exp1-2\\Data\\movie_score.csv')
n_users = data.User.nunique()
n_movies = data.Movie.nunique()
user_id_to_index = {user_id:index for index,user_id in enumerate(data.User.unique())}
movie_id_to_index = {movie_id:index for index,movie_id in enumerate(data.Movie.unique())}
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

tags_data = pd.read_csv('D:\\2024秋\\web\\webinfoexp\\exp1-2\\Data\\selected_movie_top_1200_data_tag.csv')
#tags_data['Tags'] = tags_data['Tags'].apply(eval)
print(tags_data)
vectorizer = TfidfVectorizer()
tags_matrix = vectorizer.fit_transform(tags_data['Tags'])
print(tags_matrix.shape)
tag_similarity = tags_matrix * tags_matrix.T
print(tag_similarity.shape)


# 找到Time列的最大值
max_time = pd.to_datetime(data['Time']).max()
min_time = pd.to_datetime(data['Time']).min()
print(max_time-min_time)
print(f"Latest timestamp in dataset: {max_time}")
train_data['Decay'] =1 ** ((max_time - pd.to_datetime(train_data['Time'])) / (max_time - min_time))
print(train_data['Decay'].values)


# 构建用户-电影评分矩阵
#训练集
train_data_matrix = np.full((n_users, n_movies), np.nan)
for line in train_data.itertuples():
    user_index = user_id_to_index[line.User]
    movie_index = movie_id_to_index[line.Movie]
    train_data_matrix[user_index, movie_index] = line.Rate
train_mask = ~np.isnan(train_data_matrix)

decay_matrix = np.full((n_users, n_movies), np.nan)
for line in train_data.itertuples():
    user_index = user_id_to_index[line.User]
    movie_index = movie_id_to_index[line.Movie]
    decay_matrix[user_index, movie_index] = line.Decay
# 将训练集中的NaN值替换为0以进行SVD分解
train_data_matrix = np.where(train_mask, train_data_matrix * decay_matrix, 0)

#测试集
test_data_matrix = np.full((n_users, n_movies), np.nan)
for line in test_data.itertuples():
    user_index = user_id_to_index[line.User]
    movie_index = movie_id_to_index[line.Movie]
    test_data_matrix[user_index, movie_index] = line.Rate
test_mask = ~np.isnan(test_data_matrix)
# 将测试集中的NaN值替换为0以进行SVD分解
test_data_matrix[np.isnan(test_data_matrix)] = 0

for k in [10,20,30,40,50,60,70,80,90,100,200,300,400,500]:
    # svd分解
    U,sigma,vt= svds(train_data_matrix,k)
    sigma_diag = np.diag(sigma)
    svd_prediction = np.dot(np.dot(U,sigma_diag),vt)
    # print(svd_prediction.shape)
    svd_prediction = np.clip(svd_prediction, 0, 5)
    svd_prediction_tags = svd_prediction*tag_similarity
    # 找出最大值并正则化到0-5范围
    max_val = np.max(svd_prediction_tags)
    svd_prediction_tags = 5 * (svd_prediction_tags / max_val)
    svd_prediction_tags = np.clip(svd_prediction_tags, 0, 5)

    print(f'k={k}次：')
    # 计算训练集的MSE误差
    # print(train_mask)
    train_mse = np.mean((svd_prediction[train_mask] - train_data_matrix[train_mask]) ** 2)
    print(f"Train MSE: {train_mse}")
    # 计算测试集的MSE误差
    test_mse = np.mean((svd_prediction[test_mask] - test_data_matrix[test_mask]) ** 2)
    print(f"Test MSE: {test_mse}")

    #增加标签辅助预测
    # 计算训练集的MSE误差
    # print(train_mask)
    train_mse = np.mean((svd_prediction_tags[train_mask] - train_data_matrix[train_mask]) ** 2)
    print(f"Train MSE with tags: {train_mse}")
    # 计算测试集的MSE误差
    test_mse = np.mean((svd_prediction_tags[test_mask] - test_data_matrix[test_mask]) ** 2)
    print(f"Test MSE with tags: {test_mse}")




