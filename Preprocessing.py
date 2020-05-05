# 1. LOAD LIBRARIES
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


# 2. IMPORT DATASET
df = pd.read_csv('data/A2Z_Insurance.csv')

# 3. CHANGE COLUMN NAMES
df.columns = ['ID', 'First_Policy', 'Birthday',
              'Education', 'Salary', 'Area',
              'Children', 'CMV', 'Claims',
              'Motor', 'Household',
              'Health', 'Life',
              'Work_Compensations']

# Due to the incoherence that the variable Birthday lead to (in more that 2000 observations), we decided to remove it.
# The reason why we do that is explained in the commented code in section '10. COHERENCE CHECKING'
df = df.drop(columns='Birthday')

# 4. METADATA
metadata = {'ID': 'Customer ID', "First_Policy": "Year of the Customer's first policy ",
            'Birthday': 'Birthday Year', 'Education': 'Academic Degree', 'Salary': 'Gross Monthly Salary (€) ',
            'Area': 'Living Area', 'Children': 'Has Children (Y=1)',
            'CMV': 'Customer Monetary Value = (annual profit from the customer) * (number of years that they are a customer) - (acquisition cost)',
            'Claims': 'Claims Rate (last 2 years) = Amount paid by the insurance company (€)/ Premiums (€)',
            'Motor': 'Premiums (€) in LOB: Motor', 'Household': 'Premiums (€) in LOB: Household',
            'Health': 'Premiums (€) in LOB: Health', 'Life': 'Premiums (€) in LOB: Life',
            'Work_Compensations': 'Premiums (€) in LOB: Work Compensations'}

# 5. MISSING VALUES TREATMENT
# 5.1 Check missings
# 5.1.1 By column
missings_column = df.isnull().sum().reset_index()
missings_column.columns = ['Variable', '# missings']

# 5.1.2 By row
missings_row = pd.DataFrame(df.isnull().sum(axis=1)).reset_index()
missings_row.columns = ['client', '# missings']
missings_row = missings_row['# missings'].value_counts().reset_index()
missings_row.columns = ['# missings', '# observations']

# 5.2 Drop missing values
df.dropna(subset=['Area'], inplace=True)
df.reset_index(drop=True, inplace=True)

# 5.3 Fill missing values with 0 (or equivalent)
# All Premiums missing values are consider not done insurances and so the are filled with 0
df['Motor'] = df['Motor'].fillna(0)
df['Health'] = df['Health'].fillna(0)
df['Life'] = df['Life'].fillna(0)
df['Work_Compensations'] = df['Work_Compensations'].fillna(0)
# Fill education with one new level of education (the lowest one, Primary)
df['Education'] = df['Education'].fillna('0 - Primary')

# 5.3 Fill missing values with most common class
df.Children.value_counts()
# Fill children with more common value
df['Children'] = df['Children'].fillna(1)

# 5.4 Temporarily median imputation
# Missing will be filled with median only too realise univariate outliers analysis.
# Thats why it is created a temporary dataframe (temp_df).
temp_df = df[
    ['ID', 'First_Policy', 'Salary', 'CMV', 'Claims', 'Motor', 'Household', 'Health', 'Life', 'Work_Compensations']]
temp_df = temp_df.fillna(temp_df.median())

# 6. DEAL WITH NEGATIVE MONETARY VALUES
# All the clients with negative values in premiums have abandoned one insurance and so money has been giving back to them.
# The new binnary variable 'Abandoned' is crated to save the information about those clients we know for sure that abandoned, at least, one insurance.
df['Abandoned'] = 0
df.loc[df['Motor'] < 0, 'Abandoned'] = 1
df.loc[df['Household'] < 0, 'Abandoned'] = 1
df.loc[df['Health'] < 0, 'Abandoned'] = 1
df.loc[df['Life'] < 0, 'Abandoned'] = 1
df.loc[df['Work_Compensations'] < 0, 'Abandoned'] = 1

# To facilitate the creation of new variables (such as "Total Premiums") in section 9 and
# the interpretability of the premiums variables, we changed all negative premiums to 0.
df.loc[df['Motor'] < 0, 'Motor'] = 0
df.loc[df['Household'] < 0, 'Household'] = 0
df.loc[df['Health'] < 0, 'Health'] = 0
df.loc[df['Life'] < 0, 'Life'] = 0
df.loc[df['Work_Compensations'] < 0, 'Work_Compensations'] = 0

# 7. OUTLIERS ANALYSIS
# 7.1 Univariate analysis
# The column Outlier in temp_df is a binnary indicating if one observationn is consider an outlier in the univariate outlier analysis
temp_df['Outlier'] = 0

# The following values that serve as the boundaries frow whom one observation is consider an outlier in each variable
# are selected through the observation of the boxplots and distplots presented in the section 7.1.1 and 7.1.2, respectively
temp_df.loc[df['First_Policy'] > 2020, 'Outlier'] = 1
a_outl = min(temp_df['Outlier'].value_counts())
temp_df.loc[df['Salary'] > 5100, 'Outlier'] = 1
b_outl = min(temp_df['Outlier'].value_counts()) - a_outl
temp_df.loc[(df['CMV'] < -1000) | (df['CMV'] > 10000), 'Outlier'] = 1
c_outl = min(temp_df['Outlier'].value_counts()) - a_outl - b_outl
temp_df.loc[df['Motor'] > 2000, 'Outlier'] = 1
d_outl = min(temp_df['Outlier'].value_counts()) - a_outl - b_outl - c_outl
temp_df.loc[df['Household'] > 4000, 'Outlier'] = 1
e_outl = min(temp_df['Outlier'].value_counts()) - a_outl - b_outl - c_outl - d_outl
temp_df.loc[df['Health'] > 5000, 'Outlier'] = 1
f_outl = min(temp_df['Outlier'].value_counts()) - a_outl - b_outl - c_outl - d_outl - e_outl
temp_df.loc[df['Work_Compensations'] > 1500, 'Outlier'] = 1
g_outl = min(temp_df['Outlier'].value_counts()) - a_outl - b_outl - c_outl - d_outl - e_outl - f_outl

outliers_vis = pd.DataFrame(
    {'VARIABLE': ['First_Policy', 'Salary', 'CMV', 'Motor', 'Household', 'Health', 'Work_Compensations'],
     '# outliers': [a_outl, b_outl, c_outl, d_outl, e_outl, f_outl, g_outl]})
ax = outliers_vis.sort_values(by='# outliers', ascending=False).plot.bar(x='VARIABLE', y='# outliers')
for p in ax.patches: ax.annotate(np.round(p.get_height(), decimals=2), (p.get_x() + p.get_width() / 2., p.get_height()),
                                 ha='center', va='center', xytext=(0, 10), textcoords='offset points')
ax.set_xlabel('VARIABLE', size=12)
ax.set_ylabel('Number of outliers', size=12)
plt.xticks(fontsize=11)
# plt.tight_layout()

# create temporary df for multivariate outliers analysis with the column 'Outlier' set to zero in all obs
temp_multi = temp_df
# This dataframe 'outliers' will be use in the future in section 19., and it needed to be imported here, for now the next 3 lines of code can be ignored
outliers = temp_df.loc[temp_df['Outlier'] == 1]
outliers = df.loc[df['ID'].isin(outliers['ID'])]  # remove outliers from original dataset

# remove outliers from temporary df to gain interpretability in the following graphs (6.1.1 and 6.1.2)
# once this are plotted without including the observations consider as outliers in univariate analysis
temp_df = temp_df.loc[temp_df['Outlier'] == 0]

# 7.1.1 Boxplot visualization
f, axes = plt.subplots(3, 3, figsize=(4, 2))
plt.subplots_adjust(wspace=0.3, hspace=0.3)
sb.boxplot(temp_df["First_Policy"], color=sb.color_palette("Blues")[1], ax=axes[0, 0])
sb.boxplot(temp_df["Salary"], color=sb.color_palette("Blues")[3], ax=axes[0, 1])
sb.boxplot(temp_df["CMV"], whis=5, color=sb.color_palette("Blues")[5], ax=axes[0, 2])
sb.boxplot(temp_df["Claims"], color=sb.color_palette("BuGn_r")[4], ax=axes[1, 0])
sb.boxplot(temp_df["Motor"], color=sb.color_palette("BuGn_r")[3], ax=axes[1, 1])
sb.boxplot(temp_df["Household"], whis=7, color=sb.color_palette("BuGn_r")[0], ax=axes[1, 2])
sb.boxplot(temp_df["Health"], whis=2.5, color=sb.cubehelix_palette(8)[2], ax=axes[2, 0])
sb.boxplot(temp_df["Life"], whis=7.5, color=sb.cubehelix_palette(8)[4], ax=axes[2, 1])
sb.boxplot(temp_df["Work_Compensations"], whis=7, color=sb.cubehelix_palette(8)[6], ax=axes[2, 2])

# 7.1.2 Histogram visualization
f, axes = plt.subplots(3, 3, figsize=(7, 7))
plt.subplots_adjust(wspace=0.3, hspace=0.3)
sb.distplot(temp_df["First_Policy"], color=sb.color_palette("Blues")[1], ax=axes[0, 0])
sb.distplot(temp_df["Salary"], color=sb.color_palette("Blues")[3], ax=axes[0, 1])
sb.distplot(temp_df["CMV"], color=sb.color_palette("Blues")[5], ax=axes[0, 2])
sb.distplot(temp_df["Claims"], color=sb.color_palette("BuGn_r")[4], ax=axes[1, 0])
sb.distplot(temp_df["Motor"], color=sb.color_palette("BuGn_r")[3], ax=axes[1, 1])
sb.distplot(temp_df["Household"], color=sb.color_palette("BuGn_r")[0], ax=axes[1, 2])
sb.distplot(temp_df["Health"], color=sb.cubehelix_palette(8)[2], ax=axes[2, 0])
sb.distplot(temp_df["Life"], color=sb.cubehelix_palette(8)[4], ax=axes[2, 1])
sb.distplot(temp_df["Work_Compensations"], color=sb.cubehelix_palette(8)[6], ax=axes[2, 2])

# 7.2 Multivariate analysis
# 7.1.2 K-means
std_data = stats.zscore(
    temp_multi[['First_Policy', 'Salary', 'CMV', 'Motor', 'Household', 'Health', 'Life', 'Work_Compensations']])
k_means = KMeans(n_clusters=70, init='k-means++', n_init=50, max_iter=300).fit(std_data)

# Clusters table has the cluster that each observation belongs to ('Cluster'), the original ID of the variables ('ID') and
# a binnary ('Outlier') indicating if the clients were consider outliers in the univariate analysis.
clusters = pd.DataFrame(k_means.labels_, columns=['Cluster'])
clusters['ID'] = temp_multi['ID']
clusters['Outlier'] = temp_multi['Outlier']

clusters.Cluster.value_counts()
outliers_multi = clusters.groupby(by='Cluster')['Cluster'].count().reset_index(name='N')

# We consider multivariate outliers the clusters with less than 7 observations.
outliers_multi = outliers_multi.loc[outliers_multi['N'] <= 6]
outliers_multi = pd.merge(clusters, outliers_multi, on='Cluster', how='inner').drop(columns='N')

# In this way, the observations belonging to the 'outliers_multi' and with value 1 in the 'Outlier' column will be excluded from the analysis
df = df.loc[~df['ID'].isin(outliers['ID'])]  # remove outliers from original dataset

# 5. MISSING VALUES TREATMENT (CONTINUATION)
# 5.5 Mean imputation based on education levels
Salary_mean = dict(df.groupby('Education')['Salary'].mean().round())

df.loc[(df['Education'] == '0 - Primary') & (df['Salary'].isnull()), 'Salary'] = Salary_mean['0 - Primary']
df.loc[(df['Education'] == '1 - Basic') & (df['Salary'].isnull()), 'Salary'] = Salary_mean['1 - Basic']
df.loc[(df['Education'] == '2 - High School') & (df['Salary'].isnull()), 'Salary'] = Salary_mean['2 - High School']
df.loc[(df['Education'] == '3 - BSc/MSc') & (df['Salary'].isnull()), 'Salary'] = Salary_mean['3 - BSc/MSc']
df.loc[(df['Education'] == '4 - PhD') & (df['Salary'].isnull()), 'Salary'] = Salary_mean['4 - PhD']

# 5.6 Predict missing values through neighbors - K Nearest Neighbors
my_data_to_reg = df[['ID', 'CMV', 'Claims', 'Household', 'First_Policy']]
my_data_to_reg = my_data_to_reg.set_index('ID')
my_data_to_reg_incomplete = my_data_to_reg[my_data_to_reg.First_Policy.isna()]
my_data_to_reg_complete = my_data_to_reg[~my_data_to_reg.index.isin(my_data_to_reg_incomplete.index)]

# Defining the parameters of the KNeighborsRegressor
my_regressor = KNeighborsRegressor(10, weights='distance', metric='euclidean')

# Applying the regressor to 'first policy'
neigh = my_regressor.fit(my_data_to_reg_complete.loc[:, ['CMV', 'Claims', 'Household']],
                         my_data_to_reg_complete.loc[:, ['First_Policy']])
imputed_Policy = neigh.predict(my_data_to_reg_incomplete.drop(columns=['First_Policy']))
temp_missing = pd.DataFrame(imputed_Policy.reshape(-1, 1), columns=['First_Policy'])
my_data_to_reg_incomplete = my_data_to_reg_incomplete.drop(columns=['First_Policy'])
my_data_to_reg_incomplete = my_data_to_reg_incomplete.reset_index()
my_data_to_reg_incomplete = pd.concat([my_data_to_reg_incomplete, temp_missing], axis=1)
# Round years
my_data_to_reg_incomplete['First_Policy'] = my_data_to_reg_incomplete['First_Policy'].round()

# Join new values to the original dataset
df = df.set_index('ID')
my_data_to_reg_incomplete = my_data_to_reg_incomplete.set_index('ID')
df['First_Policy'] = df['First_Policy'].fillna(my_data_to_reg_incomplete['First_Policy'])
df.reset_index(inplace=True)

# 8. GET INSIGHTS
# See variables types and descritive statistics
df.dtypes
insights = df.describe()
numerical = df[
    ['First_Policy', 'Salary', 'CMV', 'Claims', 'Motor', 'Household', 'Health', 'Life', 'Work_Compensations']]

# gain insights related with correlations through the the heatmap relative to the correlation matrix
corr_matrix = numerical.corr()
mask = np.zeros_like(corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sb.heatmap(data=corr_matrix, mask=mask, center=0, annot=True, linewidths=2, cmap='coolwarm')
# plt.tight_layout()

# 9. TRANSFORM VARIABLES
# 9.1 Create new variables
df['Education_Class'] = df['Education'].str[:1]
df['Education'] = df['Education'].str[4:]
df['Higher_Education'] = np.where((df['Education'] == 'BSc/MSc') | (df['Education'] == 'PhD'), 1, 0)
# Transform the ordinal variable education in a numeric one
df.loc[df['Education'] == 'Primary', 'Years_Education'] = 5
df.loc[df['Education'] == 'Basic', 'Years_Education'] = 9
df.loc[df['Education'] == 'High School', 'Years_Education'] = 12
df.loc[df['Education'] == 'BSc/MSc', 'Years_Education'] = 16
df.loc[df['Education'] == 'PhD', 'Years_Education'] = 20
# df['Age'] = 2016 - df['Birthday']
# Change 'First_Policy' to facilitate interpretability
df['Customer_Years'] = 2016 - df['First_Policy']
df['Annual_Profit'] = df['CMV'] / df['Customer_Years']
df['Total_Premiums'] = df['Motor'] + df['Household'] + df['Health'] + df['Life'] + df['Work_Compensations']
df['Salary_Invested'] = df['Total_Premiums'] / (df['Salary'] * 12) * 100
# Create relative premiums
df['Motor_Share'] = df['Motor'] / df['Total_Premiums'] * 100
df['Household_Share'] = df['Household'] / df['Total_Premiums'] * 100
df['Health_Share'] = df['Health'] / df['Total_Premiums'] * 100
df['Life_Share'] = df['Life'] / df['Total_Premiums'] * 100
df['Work_Share'] = df['Work_Compensations'] / df['Total_Premiums'] * 100
df = df.fillna(0)

# 9.2 Change data types
df['Children'] = df['Children'].astype('int64')
df['First_Policy'] = df['First_Policy'].astype('int64')
df['Years_Education'] = df['Years_Education'].astype('int64')
df['Area'] = df['Area'].astype('int64')

# 10. COHERENCE CHECKING
df['Incoherent'] = 0
# df.loc[(df['Birthday'] > df['First_Policy']) | (df['First_Policy']>2016), 'Incoherent'] = 1
# df.loc[(df['Birthday']>2016) | (df['Birthday']<1900) , 'Incoherent'] = 1
df.loc[(df['Salary'] < 0), 'Incoherent'] = 1
df.loc[(df['Area'] < 1) | (df['Area'] > 4), 'Incoherent'] = 1
df.loc[(df['Children'] > 1) | (df['Children'] < 0), 'Incoherent'] = 1
# df.loc[df['Age']<=df['Years_Education']+4, 'Incoherent'] = 1
# df.loc[df['Age']<18, 'Incoherent'] = 1

# Check number of incoherences
df['Incoherent'].value_counts()
# Remove incoherences
df = df.loc[df['Incoherent'] == 0]

# 11. DATA PARTITION AND STANDARDIZATION
# 11.1 Data Partition into subsets

# Separate df into 'Value' of the customers and channel of 'Consumption'
Consumption = df[['ID', 'Motor_Share', 'Household_Share', 'Health_Share', 'Life_Share', 'Work_Share']].set_index('ID')
Value = df[
    ['ID', 'Customer_Years', 'Years_Education', 'Salary_Invested', 'Area', 'Children', 'CMV', 'Abandoned']].set_index(
    'ID')

# 11.1.1 Correlations into Consumpiton
# Verifing the correlations into 'Consumption' to help in the selection of the variables
corr_consump = Consumption.corr()
mask = np.zeros_like(corr_consump, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sb.heatmap(data=corr_consump, mask=mask, center=0, annot=True, linewidths=2, cmap='coolwarm')
# plt.tight_layout()

# 11.1.2 Correlations into Value
# Verify correlations between the numerical variables in 'Value' to help in the selection of the variables
num = df[['ID', 'Customer_Years', 'Years_Education', 'Salary_Invested', 'CMV']].set_index('ID')
corr_value = num.corr()
mask = np.zeros_like(corr_value, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sb.heatmap(data=corr_value, mask=mask, center=0, annot=True, linewidths=2, cmap='coolwarm')
# plt.tight_layout()

# 11.2. Data Standardization
# 11.2.1 Standardize Consumption
scaler = StandardScaler()
std_cons = scaler.fit_transform(Consumption)
std_cons = pd.DataFrame(std_cons, columns=Consumption.columns).set_index(Consumption.index)

# 11.2.2 Standardize Value
Value_non_discrete = Value[['Years_Education', 'Salary_Invested', 'CMV']]
std_value = scaler.fit_transform(Value_non_discrete)
std_value = pd.DataFrame(std_value, columns=Value_non_discrete.columns).set_index(Value_non_discrete.index)

#11.3. Export to csv

#Standardized DF
std_cons.to_csv('data/Consumption_scaler.csv')
std_value.to_csv('data/Value_scaler.csv')

#Cleaned DF partitions
Consumption.to_csv('data/Consumption_clean.csv')
Value.to_csv('data/Value_clean.csv')

#Outliers to be later predicted
outliers.to_csv('data/Outliers.csv')

#DF cleaned
df.to_csv('data/df.csv')