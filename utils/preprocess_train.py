import numpy as np
import pandas as pd
from IPython import embed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.preprocessing import Imputer


prefix = "./data/all_data/"
print("starting")
all_data = pd.read_csv(prefix + "train_data.csv", quotechar='"', skipinitialspace=True, index_col=False, encoding="ISO-8859-1")
user_data = pd.read_csv(prefix + "users_metadata.csv", quotechar='"', skipinitialspace=True, index_col=False, encoding="ISO-8859-1")
video_data = pd.read_csv(prefix + "videos_metadata.csv", quotechar='"', skipinitialspace=True, index_col=False, encoding="ISO-8859-1")

video_id = all_data["video_id"]
rel_1_data = all_data[["video_id", "uploader", "age", "category", "length", "views", "rate", "ratings", "comments"]]
print("joining_1")
rel_2_data = pd.merge(rel_1_data, user_data, on='uploader', how='left')
print("joining_2")
rel_3_data = pd.merge(rel_2_data, video_data, on='video_id', how='left')


print("adding extra columns - 1")
s1 = pd.Series(rel_3_data["category"]).to_sparse()
rel_5_data = pd.concat([rel_3_data, pd.get_dummies(s1)], axis=1)
rel_5_data = rel_5_data.ix[:, 2:]
rel_5_data = rel_5_data.drop(['category'], axis=1)


imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(rel_5_data)
rel_5_data.to_csv('train_joined_data.csv', sep=',', index=False, encoding='utf-8')

# Get number of components
X = imp.fit_transform(rel_5_data)
X_scaled = preprocessing.scale(X)
X_std = StandardScaler().fit_transform(X_scaled)


mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
# print('Covariance matrix \n%s' %cov_mat)

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)


# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)


tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)


for i in range(0,len(cum_var_exp)):
    if np.real(cum_var_exp[i]) > 85:
        num_comp = i + 1
        break

print("num_comp", num_comp)

# WORKING PCA
print("starting PCA")
sklearn_pca = PCA(n_components=num_comp)
Y_sklearn = sklearn_pca.fit_transform(X_std)


scaled_features = StandardScaler().fit_transform(X)
scaled_features_df = pd.DataFrame(Y_sklearn, index=rel_5_data.index)

final_df = pd.concat([video_id, scaled_features_df], axis=1)
final_df.to_csv('train_preprocessed_data.csv', sep=',', index=False, encoding='utf-8')
print("done")
