import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import surprise 
from surprise  import SVD, Dataset, KNNBasic, KNNWithMeans, accuracy
from surprise.model_selection import cross_validate, KFold

def eval(splits, model):
    running_rmse = 0
    running_mae = 0
    cnt = 0
    for i, (train, test) in enumerate(splits):
        model.fit(train)
        pred = model.test(test)

        running_rmse += accuracy.rmse(pred, verbose=False)
        running_mae += accuracy.mae(pred, verbose=False)
        cnt += 1
    
    running_mae /= cnt
    running_rmse /= cnt
    
    return running_mae, running_rmse

reader = surprise.Reader(rating_scale=(1, 5))
dataframe = pd.read_csv('./ratings.csv').drop(['timestamp'], axis=1)
dataframe = dataframe[dataframe.userId < 1200]

data = Dataset.load_from_df(dataframe, reader)

#reader = surprise.Reader(rating_scale=(1, 5), skip_lines=1, line_format='user item rating timestamp', sep=',')
#data = Dataset.load_from_file('./ratings_small.csv', reader) 

kf = KFold(n_splits=5)

# User-based CF
#print(" --- User CF --- ")
#print("   -- Cosine Correlation --")
#user_CF = KNNBasic(k=5, sim_options={'name': 'pearson', 'user_based': False}, verbose=True)
#mae, rmse = eval(kf.split(data), user_CF)
#print("  MAE =", mae, "RMSE =", rmse)


# Test User-based CF
print(" --- User CF --- ")
print("   -- Cosine Correlation --")
user_CF = KNNBasic(k=40, sim_options={'name': 'cosine', 'user_based': True}, verbose=True)
cross_validate(user_CF, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("")

print("   -- Pearson Correlation --")
user_CF = KNNWithMeans(k=40, sim_options={'name': 'pearson', 'user_based': True}, verbose=True)
cross_validate(user_CF, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("")

print("   -- MSD Correlation --")
user_CF = KNNBasic(k=40, sim_options={'name': 'msd', 'user_based': True}, verbose=True)
cross_validate(user_CF, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("")


# Test Item-based CF
print(" --- Item CF --- ")
print("   -- Pearson Correlation --")
user_CF = KNNBasic(k=40, sim_options={'name': 'pearson', 'user_based': False}, verbose=True)
cross_validate(user_CF, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("")

print("   -- Cosine Correlation --")
user_CF = KNNBasic(k=40, sim_options={'name': 'cosine', 'user_based': False}, verbose=True)
cross_validate(user_CF, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("")

print("   -- MSD Correlation --")
user_CF = KNNBasic(k=40, sim_options={'name': 'msd', 'user_based': False}, verbose=True)
cross_validate(user_CF, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("")

# Test Probablistic Matrix Factorization
print(" --- PMF --- ")
PMF = SVD(biased=False)
cross_validate(PMF, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("")

user_scores = []
item_scores = []

# Test various numbers of neighbors for both user-based and item-based correlation
for k in range(1, 81):
    user_CF = KNNBasic(k=k, sim_options={'name': 'pearson', 'user_based': True}, verbose=False)
    mae, rmse = eval(kf.split(data), user_CF)
    print("User-based, k =",k,", RMSE =", rmse)
    user_scores.append(rmse)
    
    user_CF = KNNBasic(k=k, sim_options={'name': 'pearson', 'user_based': False}, verbose=False)
    mae, rmse = eval(kf.split(data), user_CF)
    print("Item-based, k =",k,", RMSE =", rmse)
    item_scores.append(rmse)


scores = []
for i in range(len(user_scores)):
    scores.append([user_scores[i], item_scores[i]])

scores = np.asarray(scores)
np.savetxt("rmse_over_k.csv", scores, delimiter=",")
