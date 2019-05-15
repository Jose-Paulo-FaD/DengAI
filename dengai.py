import utils
import matplotlib.pyplot as plt
import pandas as pd

training_features_url = 'https://raw.githubusercontent.com/Jose-Paulo-FaD/DengAI/master/dengue_features_train.csv'
training_labels_url = 'https://raw.githubusercontent.com/Jose-Paulo-FaD/DengAI/master/dengue_labels_train.csv'

# Import data
feature_frame = pd.read_csv(training_features_url)
labels_frame = pd.read_csv(training_labels_url)

# Drop redundant columns
labels_frame.drop(['city', 'year', 'weekofyear'], axis = 1, inplace = True)

# Join features and labels
all_frame = feature_frame.join(labels_frame)

# Separate San Juan and Iquitos
sj_frame = all_frame.loc[all_frame['city'] == 'sj'].reset_index(drop = True)
iq_frame = all_frame.loc[all_frame['city'] == 'iq'].reset_index(drop = True)

# Plot correlation matrix, sorted by total cases correlation
plt.figure(figsize = (8, 8))
utils.corrplot(all_frame.corr().sort_values('total_cases', axis = 1), size_scale = 150)

plt.show()