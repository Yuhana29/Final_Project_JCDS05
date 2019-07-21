# DATAFRAME

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
warnings.filterwarnings("ignore")

#==================================================== LOAD DATA ==============================================================================================================================================

df = pd.read_csv('train.csv')


# print(train.isnull().sum())
# print(df.head())

# print(df.info())

# print(df.describe())
# print(df.columns)
#============================================ DATA VISUALIZATION & ANALYSIS ===================================================================================================================================================

# find the best features to be used for machine learning

from sklearn.feature_selection import SelectKBest, chi2


x = df.iloc[:,0:20]                     # without price_range
y = df.iloc[:,-1]                       # only price_range

# print(x)
# print(y)

#==========================================================================================================================================

# SIGNIFICANCE OF EACH FEATURES TO PRICE RANGE

#========================================================================================================================================== 

best5features = SelectKBest(score_func= chi2, k= 10)
fit = best5features.fit(x,y)

score_feature = pd.DataFrame(fit.scores_)
# print(score_feature)

column_feature = pd.DataFrame(x.columns)
# print(column_feature)

df_score = pd.concat([score_feature, column_feature], axis = 1)
df_score.columns = ['score', 'features']
# print(df_score.nlargest(20, 'score'))

#           score       features
# 13  931267.519053            ram
# 11   17363.569536      px_height
# 0    14129.866576  battery_power
# 12    9810.586750       px_width
# 8       95.972863      mobile_wt
# 6       89.839124     int_memory
# 15      16.480319           sc_w
# 16      13.236400      talk_time
# 4       10.135166             fc
# 14       9.614878           sc_h
# 10       9.186054             pc
# 9        9.097556        n_cores
# 18       1.928429   touch_screen
# 5        1.521572         four_g
# 7        0.745820          m_dep
# 1        0.723232           blue
# 2        0.648366    clock_speed
# 3        0.631011       dual_sim
# 19       0.422091           wifi
# 17       0.327643        three_g


# from above, we can see the features with significance scores >> 1000:
# ram, px_height, battery_power, px_width
dfs = df_score.nlargest(20, 'score')
dfs1000 = dfs.iloc[:4, :2]

colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'gray', 'yellow', 'pink', 'black', 'darkblue']
plt.figure(figsize=(13,9))
plt.subplot(121)
plt.bar(dfs1000['features'], dfs1000['score'], color=colors)
plt.xticks(rotation=90)
plt.xlabel('features')
plt.ylabel('score above 1000')
plt.title('Significance above 1000')
plt.grid(True)

# we might also want to check the features with significance above > 10:
# fc, talk_time, sc_w, int_memory, and mobile_wt
dfs10 = dfs.iloc[4:9, :2]
plt.subplot(122)
plt.bar(dfs10['features'], dfs10['score'], color=colors)
plt.xticks(rotation=90)
plt.xlabel('features')
plt.ylabel('score above 10')
plt.title('Significance above 10')
plt.grid(True)

plt.tight_layout()
plt.show()


# but we still need to see the importance of each feature

#==========================================================================================================================================

# IMPORTANCE EACH FEATURES TO PRICE RANGE

#========================================================================================================================================== 


# Feature importance of each feature
# the higher the score more important or relevant is the feature towards your output variable

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(x,y)

# print(model.feature_importances_)

feature_imp = pd.Series(model.feature_importances_, index = x.columns)
# print(feature_imp.nlargest(20))

# ram              0.345984
# battery_power    0.057326
# px_height        0.048093
# px_width         0.046027
# mobile_wt        0.040421
# m_dep            0.040419
# pc               0.039939
# n_cores          0.039515
# int_memory       0.038876
# clock_speed      0.038807
# sc_h             0.038572
# talk_time        0.038162
# sc_w             0.036062
# fc               0.035924
# dual_sim         0.023516
# blue             0.022075
# wifi             0.020556
# touch_screen     0.018924
# four_g           0.017308
# three_g          0.013493
# dtype: float64

# plot it to figures:

colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'gray', 'yellow', 'pink', 'black', 'darkblue']
featureimp_fig = feature_imp.nlargest(20)
featureimp_fig.plot(kind='barh', figsize=(8, 10), color = colors, zorder=2, width=0.85)
plt.xlabel('Frequency')

plt.show()

# from above, we can see the features with the high importance:
# ram, battery_power, px_height, px_width, mobile_wt, and m_dep

# next, we will see the correlation of each feature to the price_range

#==========================================================================================================================================

# CORRELATION OF EACH FEATURES WITH PRICE RANGE 

#========================================================================================================================================== 

# Correlation can be positive (increase in one value of feature increases the value of the target variable) 
# or negative (increase in one value of feature decreases the value of the target variable)

corr_map = df.corr()
feat = corr_map.index
# print(feat)
plt.figure(figsize = (20,20))
sb.heatmap(df[feat].corr(method = 'spearman') , annot =True , cmap = 'BuPu', vmax=.8, square=True, fmt='.1f')
plt.show()

# a little different from before, the highest correlation are:
# 1. RAM: with the highest correlation (0,9)
# 2. fc: with the correlation (0,7)
# 3. pc: with the correlation (0,7)

# meanwhile another features from before:
# battery_power with the correlation (0,2) 
# px_height with the correlation (0,1) 
# px_width with the correlation (0,2)

# for further consideration of what features to be used for machine learning, we need to see further relationship from:
# ram: because it has very very high score on each significance, importance, and correlation
# px_height: because it has the high score on significance and importance eventhough it has low score on correlation
# px_width: because it has the high score on significance and importance eventhough it has low score on correlation
# battery_power: because it has the high score on significance and importance eventhough it has low score on correlation
# fc: because it has the high score on correlation eventhough in significance and importance score is low, but it still has the impact to price range
# pc: because it has the high score on correlation eventhough in significance and importance score is low, but it still has the impact to price range


#================================================================================================================================================================================== 

# RELATIONSHIP BETWEEN SELECTED FEATURES WITH PRICE RANGE

#======================================================================================================================================================================================


# =========================== RAM 

df_price0 = df[df['price_range'] == 0]
df_price1 = df[df['price_range'] == 1]
df_price2 = df[df['price_range'] == 2]
df_price3 = df[df['price_range'] == 3]

plt.subplot(221)
df_price0['ram'].hist(alpha=0.5,color='blue',label='RAM')
plt.xlabel('RAM (MB)')
plt.ylabel('Frequency')
plt.legend()
plt.title('RAM in Low Cost Category')

plt.subplot(222)
df_price1['ram'].hist(alpha=0.5,color='red',label='RAM')
plt.xlabel('RAM (MB)')
plt.ylabel('Frequency')
plt.legend()
plt.title('RAM in Medium Cost Category')

plt.subplot(223)
df_price2['ram'].hist(alpha=0.5,color='green',label='RAM')
plt.xlabel('RAM (MB)')
plt.ylabel('Frequency')
plt.legend()
plt.title('RAM in High Cost Category')

plt.subplot(224)
df_price3['ram'].hist(alpha=0.5,color='yellow',label='RAM')
plt.xlabel('RAM (MB)')
plt.ylabel('Frequency')
plt.legend()
plt.title('RAM in Very High Cost Category')

sb.jointplot("ram", "price_range", data=df, kind='kde')
plt.xlabel('RAM (MB)')

plt.show()
# =========================== Front Camera and Primary Camera

plt.figure(figsize=(15,10))
plt.subplot(221)
df['fc'].hist(alpha=0.5,color='blue',label='Front camera')
plt.legend()
plt.xlabel('MegaPixels')
plt.ylabel('Frequency')

plt.subplot(222)
df['pc'].hist(alpha=0.5,color='red',label='Primary camera')
plt.legend()
plt.xlabel('MegaPixels')
plt.ylabel('Frequency')

plt.subplot(223)
sb.swarmplot(x='price_range',y='fc',data=df,hue='price_range')

plt.subplot(224)
sb.swarmplot(x='price_range',y='pc',data=df,hue='price_range')

plt.show()

# =========================== Battery Power 
plt.figure(figsize=(13,9))
plt.subplot(121)
sb.boxplot(x="price_range", y="battery_power", data=df)
plt.subplot(122)
sb.swarmplot(x='price_range',y='battery_power',data=df,hue='price_range')

plt.show()

# =========================== Pixel Resolution Height & Pixel Resolution Width
plt.figure(figsize=(13,9))
plt.subplot(121)
sb.swarmplot(x='price_range',y='px_height',data=df,hue='price_range')

plt.subplot(122)
sb.swarmplot(x='price_range',y='px_width',data=df,hue='price_range')


plt.show()

 



