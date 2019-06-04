import utils
import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import tree
from sklearn import metrics

training_features_path = './data/dengue_features_train.csv'
training_labels_path = './data/dengue_labels_train.csv'

# Import data
feature_frame = pd.read_csv(training_features_path)
labels_frame = pd.read_csv(training_labels_path)

# Drop redundant columns
labels_frame.drop(['city', 'year', 'weekofyear'], axis = 1, inplace = True)

# Join features and labels
all_frame = feature_frame.join(labels_frame).drop(['week_start_date'], axis = 1)

# Separate San Juan and Iquitos
sj_frame = all_frame.loc[all_frame['city'] == 'sj'].reset_index(drop = True)
iq_frame = all_frame.loc[all_frame['city'] == 'iq'].reset_index(drop = True)

# Plot correlation matrix, sorted by total cases correlation
#plt.figure(figsize = (8, 8))
#utils.corrplot(all_frame.corr().sort_values('total_cases', axis = 1), size_scale = 150)

ratio = 0.75
n = sj_frame.shape[0]

y_train = sj_frame.iloc[:int(ratio*n), -1].to_frame()
X_train = sj_frame.iloc[:int(ratio*n), 1:-1]

y_test = sj_frame.iloc[int(ratio*n):, -1].to_frame()
X_test = sj_frame.iloc[int(ratio*n):, 1:-1]

method = 'mean'
normalize = True

y_train_p, _ = preprocessing.preprocess(y_train, y_train, method = method, normalize = normalize)
X_train_p, _ = preprocessing.preprocess(X_train, X_train, method = method, normalize = normalize)

y_test_p, _ = preprocessing.preprocess(y_test, y_train, method = method, normalize = normalize)
X_test_p, _ = preprocessing.preprocess(X_test, X_train, method = method, normalize = normalize)

#ts = statsmodels.tsa.seasonal.seasonal_decompose(y_test_p, freq = 52)
#ts.plot()
#plt.show()

#model = linear_model.LinearRegression()
model = tree.DecisionTreeRegressor()
model.fit(X_train_p, y_train_p)

y_hat = model.predict(X_test_p)
y_hat[y_hat < 0] = 0
#y_hat = utils.moving_average(y_hat, 9)
#y_test_p = y_test_p[4:-4]

plt.subplot(1, 2, 1)
plt.plot(np.array(y_train_p), 'ob', label = 'actual')

plt.subplot(1, 2, 2)

plt.plot(np.array(y_test_p), 'ob', label = 'actual')
plt.plot(y_hat, '^r', label = 'predicted')

plt.title('MAE: {} | Method: {} | Normalize: {}'.format(str(metrics.mean_absolute_error(y_test_p, y_hat)), str(method), str(normalize)))

plt.legend(loc='upper right')

plt.show()