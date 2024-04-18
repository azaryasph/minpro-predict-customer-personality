# %% [markdown]
# # Predict Customer Personality to Boost Marketing Campaign by Using Machine Learning
# 
# - Name : Azarya Yehezkiel Pinondang Sipahutar
# 
# **Project Goals**<br> Segment our customer base using clustering techniques. This segmentation will enable our marketing team to tailor their strategies to the specific needs, behaviors, and preferences of each customer group. By doing so, we hope to increase customer engagement, boost revenue, and reduce marketing costs by focusing our efforts where they are most likely to have an impact.<br><br>
# **Objective** <br>The objective of this project is to develop a KMeans Clustering Model. This model will analyze our customer data and identify distinct clusters based on various customer attributes. The resulting clusters will provide a clearer understanding of our diverse customer base, allowing us to target our marketing efforts more effectively.

# %% [markdown]
# ## Task 1 : Conversion Rate Analysis Based On Income, Spending And Age
# Goals : Find a pattern of Customer behavior.<br><br>
# Objective : 
# - Feature engineering 
# - Exploratory Data Analysis (EDA)  
# - Analyze Conversion Rate with other variables such as age, income, expenses, etc

# %% [markdown]
# ### Import Library

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ### Load Data & Preliminary Analysis

# %%
pd.set_option('display.max_columns', None)
df = pd.read_csv('../data/marketing_campaign_data.csv')
display(df.sample(4))

# Display the information about the DataFrame
print("DataFrame Information:")
df.info()

# Create a DataFrame for the description
desc_df = df.describe().transpose()

# Add the number of unique values to the description DataFrame
desc_df['unique'] = df.nunique()

# Display the description DataFrame
print("\nDataFrame Description:")
display(desc_df)

# %% [markdown]
# Upon initial inspection of the dataset, we have identified several key points that will influence our data preprocessing steps:
# 
# 1. **Missing Values:** The dataset contains missing values that need to be handled. Depending on the nature and amount of missing data, we may choose to fill these with appropriate values or drop the rows/columns with missing data.
# 
# 2. **Outliers:** Some features in the dataset exhibit outliers. These can significantly skew our statistical analysis and machine learning model performance. We will need to identify these outliers and decide on the best strategy to handle them, such as capping, transforming, or removing them.
# 
# 3. **Redundant Index Column:** The `Unnamed: 0` feature appears to be an index column. Since Pandas DataFrames automatically provide an index, this column is redundant and will be dropped during preprocessing.
# 
# 4. **Non-informative Columns:** The `Z_CostContact` and `Z_Revenue` features only contain a single unique value. These features do not provide any variability or valuable information for our analysis or predictive modeling, and will therefore be dropped during preprocessing.

# %% [markdown]
# ### Feature Engineering
# 
# In this section, we create new features to better understand our customers and their behaviors. Here's a brief explanation of each new feature:
# 
# 1. **Age**: This feature represents the age of each customer. It is calculated by subtracting the `Year_Birth` feature from the current year.
# 
# 2. **AgeGroup**: This feature categorizes customers into different age groups for easier analysis. The age groups are determined based on the customer's `Age` range, as suggested by this [article](https://www.researchgate.net/figure/Age-intervals-and-age-groups_tbl1_228404297). The minimum age in this dataset is 28.
# 
# 3. **Parent**: This feature indicates the parental status of each customer. It is created based on whether a customer has a kid at home or not.
# 
# 4. **NumChild**: This feature represents the total number of children each customer has. It is calculated from the sum of the `KidHome` and `TeenHome` features.
# 
# 5. **TotalAcceptedCmp**: This feature represents the total number of campaigns each customer accepted after the campaign was carried out. It is calculated from the sum of the `AcceptedCmp1` to `AcceptedCmp5` features.
# 
# 6. **TotalSpending** : This feature represents the total spending each customer spended on our platform. It is calculated from the sum of `MntCoke`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, and `MntGoldProds` features.
# 
# 7. **Total Trx**: This feature represents the total number of transactions the customer made in our store, either offline or online. It is calculated from the `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, and `NumStorePurchases` features.
# 
# 9. **ConversionRate**: This feature represents the percentage of website visitors who complete a web purchase. It is a key metric for understanding the effectiveness of our *online sales efforts*.

# %%
# Create a copy of the original dataframe to avoid modifying the original data
dfe = df.copy()

# Calculate the age of each customer based on their year of birth
dfe['Age'] = 2024 - dfe['Year_Birth']

# Categorize customers into age groups based on their age
age_grouping = [
    (dfe['Age'] >= 60),
    (dfe['Age'] >= 40 ) & (dfe['Age'] < 60),
    (dfe['Age'] >= 28) & (dfe['Age'] < 40)
]
age_category = ['Old Adults', 'Middled-aged Adults', 'Young Adults']
dfe['AgeGroup'] = np.select(age_grouping, age_category)

# add a new 'Parent' column to the DataFrame to indicate whether a customer is a parent or not
dfe['Parent'] = dfe.apply(lambda row: 'yes' if row['Kidhome'] > 0 or row['Teenhome'] > 0 else 'no', axis=1)

# Calculate the total number of children each customer has
dfe['NumChild'] = dfe['Kidhome'] + dfe['Teenhome']

# Calculate the total number of campaigns each customer accepted
dfe['TotalAcceptedCmp'] = dfe['AcceptedCmp1'] + dfe['AcceptedCmp2'] + dfe['AcceptedCmp3'] + dfe['AcceptedCmp4'] + dfe['AcceptedCmp5']

# Calculate the total spending of each customer across all product categories
dfe['TotalSpending'] = dfe['MntCoke'] + dfe['MntFruits'] + dfe['MntMeatProducts'] + dfe['MntFishProducts'] + dfe['MntSweetProducts'] + dfe['MntGoldProds']

# Calculate the total number of transactions each customer made
dfe['TotalTrx'] = dfe['NumDealsPurchases'] + dfe['NumWebPurchases'] + dfe['NumCatalogPurchases'] + dfe['NumStorePurchases']

# Calculate the conversion rate for each customer (the number of web purchases divided by the number of web visits)
dfe['ConversionRate'] = dfe['NumWebPurchases'] / dfe['NumWebVisitsMonth']

# %% [markdown]
# ### EDA

# %% [markdown]
# #### Univariate Analysis

# %% [markdown]
# ##### Outlier Checking (Numeric)

# %%
# List of numeric features for exploratory data analysis
numeric_features = [
    'Income', 'Recency', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
    'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 
    'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Age', 
    'TotalSpending', 'TotalTrx', 'ConversionRate'
]

# Create a subplot grid for boxplots
fig, axes = plt.subplots(2, 8, figsize=(24, 8))

# Set the title and background color for the figure
fig.suptitle('Outlier Checking of Necessary Numeric Features', fontsize=16, fontweight='bold', y=1.02)
fig.set_facecolor('#E8E8E8')

# Loop over each numeric feature and create a boxplot on a separate subplot
for feature, ax in zip(numeric_features, axes.flatten()):
    # Create a boxplot for the current feature
    sns.boxplot(y=dfe[feature], ax=ax, color='#D1106F', linewidth=2.1, width=0.55, fliersize=3.5)
    
    # Set the title for the current subplot
    ax.set_title(f'Boxplot of {feature}', fontsize=14, fontweight='bold', pad=5)
    
    # Remove gridlines from the current subplot
    ax.grid(False)

# Adjust the layout to prevent overlapping of subplots
plt.tight_layout()

# %% [markdown]
# In our dataset, we have identified outliers in the following features:
# 
# 1. `Income`
# 2. `MntMeatProducts`
# 3. `MntSweetProducts`
# 4. `MntGoldProds`
# 5. `NumDealsPurchases`
# 6. `NumWebPurchases`
# 7. `NumCatalogPurchases`
# 8. `NumWebVisitsMonth`
# 9. `Age`
# 10. `TotalTrx`
# 11. `ConversionRate`
# 
# - Outliers can significantly skew the results of our data analysis and predictive modeling process. They can be caused by various factors such as measurement errors, data entry errors, or extreme variation in the data.
# 
# - In this case, we have decided to cap the outliers to the lower/upper bound. This approach involves replacing the extreme values with a specified minimum and maximum value. It is a suitable method when we don't want to lose data, but at the same time, we want to limit the effect of the extreme values.
# 
# - This method is particularly beneficial for our unsupervised machine learning model, as it can help to improve the performance of the model by reducing the impact of the outliers on the model's learning process.

# %% [markdown]
# ##### Data Distribution (Numeric)

# %%
# List of numeric features for exploratory data analysis
numeric_features = [
    'Income', 'Recency', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
    'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 
    'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Age', 
    'TotalSpending', 'TotalTrx', 'ConversionRate'
]

# Create a subplot grid for KDE plots
fig, axes = plt.subplots(2, 8, figsize=(28, 9))

# Set the title and background color for the figure
fig.suptitle('KDE plot for Necessary Features', fontsize=16, fontweight='bold', y=1.02)
fig.set_facecolor('#E8E8E8')

# Loop over each numeric feature and create a KDE plot on a separate subplot
for feature, ax in zip(numeric_features, axes.flatten()):
    # Create a KDE plot for the current feature
    sns.kdeplot(x=dfe[feature], ax=ax, color='#D1106F', linewidth=0.7, fill=True)
    
    # Set the title for the current subplot
    ax.set_title(f'Distribution of {feature}', fontsize=14, fontweight='bold', pad=5)
    
    # Remove gridlines from the current subplot
    ax.grid(False)

# Adjust the layout to prevent overlapping of subplots
plt.tight_layout()

# %% [markdown]
# In my analysis of the kernel density estimation (KDE) plot for this dataset, i observed the following:
# 
# - The distribution of most features in this dataset is positively skewed. This indicates that the majority of the values lie to the right of the mean, with a tail extending towards the left.
# 
# - In case of missing values in the data, my strategy is to fill these gaps with the median value of the respective feature. The rationale behind this choice is that the median is robust to outliers, meaning it provides a better central tendency estimate for skewed distributions compared to the mean.

# %% [markdown]
# ##### Data Distribution (Categoric)

# %%
# Create a subplot grid for pie charts
fig, axs = plt.subplots(1, 2, figsize=(24, 12))
fig.set_facecolor('#E8E8E8')

# Define color palette
palette = ['#00D19B', '#D1106F', '#25A9D9']

# Get counts of each age group
age_counts = dfe['AgeGroup'].value_counts()

# Create pie chart for age group distribution
patches, texts, autotexts = axs[0].pie(
    age_counts, 
    colors=palette, 
    autopct='%1.1f%%', 
    textprops={'size': 20}
)

# Add legend and title for the first pie chart
axs[0].legend(patches, age_counts.index, loc="best", fontsize='x-large')
axs[0].set_title(
    "Distribution of Customer by Age Group", 
    fontsize=22, 
    fontweight='bold', 
    y=1.02
)

# Get counts of each parent group
parent_counts = dfe['Parent'].value_counts()

# Update color palette for the second pie chart
palette = ['#00D19B', '#D1106F']

# Create pie chart for parent group distribution
patches, texts, autotexts = axs[1].pie(
    parent_counts, 
    colors=palette, 
    autopct='%1.1f%%', 
    textprops={'size': 20}
)

# Add legend and title for the second pie chart
axs[1].legend(patches, parent_counts.index, loc="best", fontsize='x-large')
axs[1].set_title(
    "Parent Customer Distribution", 
    fontsize=22, 
    fontweight='bold', 
    y=1.02
)

# Adjust the layout to prevent overlapping of subplots
plt.tight_layout()

# Display the figure
plt.show()

# %%
# Create a subplot grid for count plots
fig, axs = plt.subplots(1, 2, figsize=(14, 6), facecolor='#E8E8E8')

# Define a function to annotate count plots
def annotate_countplot(countplot):
    for p in countplot.patches:
        height = p.get_height()
        countplot.text(
            p.get_x() + p.get_width() / 2.,
            height + 10,
            '{:1.0f}'.format(height),
            ha="center",
            fontweight='bold'
        )

# Define color palette and order of education levels
palette = ['#D1106F', '#00D19B', '#25A9D9', '#D16F11', '#6F11D1']
edu_order = ['SMA', 'D3', 'S1', 'S2', 'S3']

# Create count plot for education level
countplot = sns.countplot(
    data=dfe, 
    x='Education', 
    hue='Education', 
    order=edu_order, 
    palette=palette, 
    ax=axs[0], 
    legend=False
)

# Annotate the count plot
annotate_countplot(countplot)

# Set the title, labels, and grid for the first subplot
axs[0].set_ylim(0, 1250)
axs[0].set_title("Customer Distribution by Education Level", fontsize=18, fontweight='bold', y=1.03)
axs[0].set_xlabel('Education Level', fontsize=12)
axs[0].set_ylabel('Count', fontsize=12)
axs[0].grid(False)

# Update color palette for the second count plot
palette = ['#D1106F', '#00D19B', '#25A9D9', '#D16F11', '#6F11D1', '#11D1D1']

# Create count plot for marital status
countplot = sns.countplot(
    data=dfe, 
    x='Marital_Status', 
    hue='Marital_Status', 
    palette=palette, 
    ax=axs[1], 
    legend=False
)

# Annotate the count plot
annotate_countplot(countplot)

# Set the title, labels, and grid for the second subplot
axs[1].set_ylim(0, 950)
axs[1].set_title("Customer Distribution by Marital Status", fontsize=18, fontweight='bold', y=1.03)
axs[1].set_xlabel('Marital Status', fontsize=12)
axs[1].set_ylabel('Count', fontsize=12)
axs[1].grid(False)

# %% [markdown]
# From our analysis of the customer data, we can draw several key insights about the demographic profile of our majority customer base:
# 
# - Age Group: The majority of our customers fall within the middle-aged group, specifically between 40 and 59 years old. This could indicate that our products or services are particularly appealing to individuals in this age range.
# 
# - Parental Status: A significant proportion of our customers have children at home. This suggests that our offerings may cater well to the needs of parents or families.
# 
# - Education Level: Most of our customers have achieved a level of education up to a Bachelor's Degree. This could reflect the affordability, accessibility, or appeal of our products or services to individuals with this level of education.
# 
# - Marital Status: The majority of our customers are married. This might indicate that our products or services are popular among couples or that they cater to the needs of married individuals.
# 
# These insights can help us better understand our customer base and tailor our marketing strategies, product development, and services to meet their needs and preferences. However, it's important to remember that these are general trends and there may be significant variation within these groups. Further segmentation and analysis could provide more nuanced insights.

# %% [markdown]
# #### Multivariate Analysis

# %%
# Create a subplot grid for scatter plots
fig, axs = plt.subplots(2, 2, figsize=(24, 16))
fig.set_facecolor('#E8E8E8')

# Plot 1: Customer Conversion Rate and Income Correlation
sns.scatterplot(x='Income', y='ConversionRate', data=dfe, color='#D1106F', ax=axs[0, 0])
axs[0, 0].set_xlim(0, 200000000)
axs[0, 0].set_ylim(0, 4.7)
axs[0, 0].axvline(x=110000000, color='b', linestyle='--')
axs[0, 0].set_title("Customer Conversion Rate and Income Correlation", fontsize=19, fontweight='bold', y=1.02)
axs[0, 0].set_xlabel('Income', fontsize=13.5)
axs[0, 0].set_ylabel('Conversion Rate', fontsize=13.5)
axs[0, 0].grid(False)

# Plot 2: Customer Income and Total Spending Correlation
sns.scatterplot(x='TotalSpending', y='Income', data=dfe, color='#D1106F', ax=axs[0, 1])
axs[0, 1].set_ylim(0, 122000000)
axs[0, 1].set_xlim(0, 2700000)
axs[0, 1].axvline(x=2540000, color='b', linestyle='--')
axs[0, 1].set_title('Customer Income and Total Spending Correlation', fontsize=17, fontweight='bold', y=1.03)
axs[0, 1].set_xlabel('Total Spending', fontsize=13.5)
axs[0, 1].set_ylabel('Income', fontsize=13.5)
axs[0, 1].grid(False)

# Plot 3: Correlation Between Conversion Rate and Total Spending
sns.scatterplot(x='TotalSpending', y='ConversionRate', data=dfe, color='#D1106F', ax=axs[1, 0])
axs[1, 0].set_ylim(0, 3.8)
axs[1, 0].set_title('Correlation Between Conversion Rate and Total Spending', fontsize=18, fontweight='bold', y=1.02)
axs[1, 0].set_xlabel('Total Spending', fontsize=13.5)
axs[1, 0].set_ylabel('Conversion Rate', fontsize=13.5)
axs[1, 0].grid(False)

# Plot 4: Correlation Between Conversion Rate and Age
sns.scatterplot(x='Age', y='ConversionRate', data=dfe, color='#D1106F', ax=axs[1, 1])
axs[1, 1].set_title('Correlation Between Conversion Rate and Age', fontsize=18, fontweight='bold', y=1.02)
axs[1, 1].set_xlabel('Age', fontsize=13.5)
axs[1, 1].set_ylabel('Conversion Rate', fontsize=13.5)
axs[1, 1].grid(False)

# Adjust the layout to prevent overlapping of subplots
plt.tight_layout()

# Display the figure
plt.show()

# %% [markdown]
# In this multivariate analysis, several key relationships between variables were observed:
# 
# 1. **Income and Conversion Rate**:<br>There is a positive correlation between income and conversion rate. This suggests that as a customer's income increases, they are more likely to complete a purchase on our web platform after visiting. This could be due to higher disposable income allowing for more flexibility in purchasing decisions.
# 
# 2. **Total Spending and Income**:<br>There is a positive correlation between total spending and income. This indicates that customers with higher incomes tend to spend more. This could be a reflection of their greater purchasing power.
# 
# 3. **Total Spending and Conversion Rate**:<br> There is a positive correlation between total spending and conversion rate. This suggests that customers who spend more are also more likely to complete a purchase after visiting our web platform. This could be due to a higher level of engagement or interest in our products or services.
# 
# 4. **Conversion Rate and Age**:<br> There is no significant correlation between conversion rate and age. This indicates that the likelihood of a customer completing a purchase after visiting our web platform does not significantly vary with age. This could suggest that our platform appeals to a wide range of age groups.
# 
# These insights can help us better understand the behavior of our customers and inform our marketing and sales strategies. However, it's important to remember that correlation does not imply causation, and further investigation may be needed to understand the underlying causes of these relationships.

# %%
# Create a 3x2 grid of subplots with a specific size and background color
fig, axs = plt.subplots(2, 3, figsize=(24, 12), facecolor='#E8E8E8')

# Define the color palette and order of age groups
palette = ['#D1106F', '#00D19B', '#25A9D9']
age_order = ['Young Adults', 'Middled-aged Adults', 'Old Adults']

# Define a function to annotate the bars in a bar plot with their height values
def annotate_barplot(barplot):
    for p in barplot.patches:
        height = p.get_height()
        barplot.text(p.get_x() + p.get_width() / 2.,
                     height + 0.01,
                     '{:1.2f}'.format(height),
                     ha="center",
                     fontweight='bold')

# Create bar plots for different metrics by age group
for i, metric in enumerate(['ConversionRate', 'TotalSpending', 'Income']):
    barplot = sns.barplot(
        data=dfe,
        x='AgeGroup',
        y=metric,
        hue='AgeGroup',
        order=age_order,
        legend=False,
        palette=palette,
        errorbar=None,
        edgecolor='black',
        ax=axs[0, i]
    )
    annotate_barplot(barplot)
    axs[0, i].set_title(f"{metric} by Age Group", fontsize=18, fontweight='bold', y=1.03)
    axs[0, i].set_xlabel('Age Group', fontsize=12)
    axs[0, i].set_ylabel(metric, fontsize=12)
    axs[0, i].grid(False)

# Create bar plots for Conversion Rate by Number of Children, Parental Status, and Education Level
for i, metric in enumerate(['NumChild', 'Parent', 'Education']):
    palette = ['#D1106F', '#00D19B', '#25A9D9', '#D16F11', '#6F11D1'][:len(dfe[metric].unique())]
    barplot = sns.barplot(
        x=metric,
        y='ConversionRate',
        hue=metric,
        data=dfe,
        legend=False,
        palette=palette,
        errorbar=None,
        edgecolor='black',
        ax=axs[1, i]
    )
    annotate_barplot(barplot)
    axs[1, i].set_title(f"Conversion Rate by {metric}", fontsize=18, fontweight='bold', y=1.03)
    axs[1, i].set_xlabel(metric, fontsize=12)
    axs[1, i].set_ylabel('Conversion Rate', fontsize=12)
    axs[1, i].grid(False)

# Adjust the layout to prevent overlapping of subplots
plt.tight_layout()

# Display the figure
plt.show()

# %% [markdown]
# In my analysis of customer behavior, i've identified several key trends:
# - The highest conversion rate is for customers in the Old Adults age group (> 59 years), and based on total spending, this Old Adults age group has the most significant spending, reaching more than 700,000. This can indicate high trust and engagement in the online shopping experience. They may be more likely to complete a purchase because they understand and feel comfortable with the process.
# - Customers who do not have children or are not parents have a higher conversion rate than customers who already have children. However, the distribution of our customers in the previous pie chart shows that most of our customers already have children or are married. Therefore, the potential for a higher conversion rate exists for customers who do not have children.
# - Customers who don't have a degree (still in high school) have the lowest conversion rate compared to customers who already have a degree.

# %%
# Define a list of numerical features for which we want to compute correlations
numerical_features = [
       'Income', 'Recency', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
       'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 
       'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Complain',
       'Response', 'Age', 'NumChild', 'TotalAcceptedCmp', 'TotalSpending', 'TotalTrx', 
       'ConversionRate'
]

# Create a new figure with a specific size and background color
plt.figure(figsize=(16, 9.5), facecolor='#E8E8E8')

# Compute the correlation matrix for the numerical features and plot it as a heatmap
correlation_matrix = dfe[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

# Set the title of the heatmap
plt.title('Correlation Heatmap', fontsize=18, fontweight='bold', y=1.02)

# Display the heatmap
plt.show()

# %% [markdown]
# Based on the correlation Heatamp above, we can see that the highest correlation is between Income and Conversion Rate. This indicates that the higher the income, the higher the conversion rate. This can be seen from the scatter plot above, where the higher the income, the higher the conversion rate.

# %% [markdown]
# ## Task 2 : Data Cleaning & Preprocessing
# Goals : Preparing raw data into clean data ready to be processed by machine learning<br><br>
# Objective : 
# - Handle Missing Values
# - Handle Duplicate Values
# - Handle Infinity values 
# - Feature Selection 
# - Feature Encoding
# - Standarization

# %% [markdown]
# #### Handle missing values

# %%
# make a copy of previous dataframe for next step (Data Preprocessing)
dfp = dfe.copy()

# Print missing values
missing_col = dfp.isna().sum()
display_missing_col = missing_col[missing_col > 0]

# Calculate percentage of missing values
missing_percentage = (dfp.isna().sum() / len(dfp)) * 100
display_missing_percentage = missing_percentage[missing_percentage > 0]

# Format the percentages
display_missing_percentage = display_missing_percentage.map('{:.2f}%'.format)

print(f'Missing Values : \n \n{display_missing_col}')
print(f'\nPercentage of Missing Values : \n \n{display_missing_percentage}')

# %% [markdown]
# Based on the percentage of missing values in two features, `Income` and `ConversionRate`, we can see that the percentage of missing values in the `Income` is not too large and the distribution of the data is positively skewed, so we can fill in the missing values with the median value. However, the percentage of missing values in the `ConversionRate` seems reasonable to drop the missing values. 

# %%
# Select the columns 'NumWebPurchases', 'NumWebVisitsMonth', and 'ConversionRate' from the dataframe 'dfp'
missing_cr = dfp[['NumWebPurchases', 'NumWebVisitsMonth', 'ConversionRate']]

# Filter the rows in 'missing_cr' where any of the columns have missing values
missing_crdf = missing_cr[missing_cr.isna().any(axis=1)]

# Print the dataframe 'missing_crdf' which contains the rows with missing values
print(f"Highlighted Missing values : \n")
display(missing_crdf)

# This statement indicates that the missing values in the 'ConversionRate' column are not missing at random. 
# This could mean that there's a specific reason or pattern behind the missing values in this column.
print('*Conversion Rate not missing at Random*')

# %% [markdown]
# The reason to drop this missing value at conversion rate is because My Conversion Rate formula is based on the number of visitors who complete a purchase after visiting our web platform. So, if the value of the conversion rate is missing, it means that the customer has not made a purchase after visiting our web platform. So, we can drop the missing values in the `ConversionRate` feature.

# %%
# print total null on income and conversion rate
total_null_income = dfp['Income'].isna().sum()
total_null_conrate = dfp['ConversionRate'].isna().sum()
print(f"Total Missing Values on Income Column = {total_null_income}")
print(f"Total Missing Values on Conversion Rate Column = {total_null_conrate}")

# print median income
median_income = dfp['Income'].median()
print(f"\nIncome Median to fill the missing value: {median_income}")

# handle missing values with fill and drop method
dfp['Income'].fillna(dfp['Income'].median(), inplace=True)
dfp.dropna(subset=['ConversionRate'], inplace=True)

# checking missing values if still exist
nonull_income = dfp['Income'].isna().sum()
nonull_conrate = dfp['ConversionRate'].isna().sum()
print(f"\nMissing Values on Income Column after handling = {nonull_income}")
print(f"Missing Values on Conversion Rate Column after handling = {nonull_conrate}")

# %% [markdown]
# Miising Values Handled respectively in `Income` and `ConversionRate` feature

# %% [markdown]
# #### No Duplicates

# %%
# Calculate the total number of duplicate rows in the dataframe 'dfp'
total_duplicate = dfp.duplicated().sum()

# Print the total number of duplicate rows
print(f"Total Duplicated Data = {total_duplicate}")

# %% [markdown]
# No duplicates in the dataset

# %% [markdown]
# #### Fix the Infinity Value On Conversion Rate Features

# %% [markdown]
# - This infinity value is caused by the number of visitors who have not made a purchase after visiting our web platform. So, we can replace the infinity value with 0 and drop the missing values in the `ConversionRate` feature. if we don't handle this infinity value, the data cannot be standardized and will cause an error in the machine learning model.

# %%
# Print count Infiinity values in dataframe
count_inf = dfp.map(lambda x: isinstance(x, float) and x == float('inf')).sum().sum()
print(f"Count of Infinity Values :\nIt Contains {str(count_inf)} Infinite values in dataframe")

# print column where infinity values exist
col_inf = dfp.columns[dfp.map(lambda x: isinstance(x, float) and x == float('inf')).any()]
print("\nColumns where Infinity values exist:")
print(", ".join(col_inf))

# Replace infinity values with NaN and drop them
dfp.replace([np.inf, -np.inf], np.nan, inplace=True)
dfp.dropna(inplace=True)

# Print the number of entries in the dataframe
print(f"\nEntries after cleaning: {len(dfp)}")

# Check if there are still any infinity values
no_inf = dfp.map(lambda x: isinstance(x, float) and x == float('inf')).sum().sum()
print(f"Infinity values remaining: {no_inf}")

# %% [markdown]
# - infinity Values handled in `ConversionRate` feature

# %% [markdown]
# #### Handle Outliers
# - Outliers are need to be handle because our task is cluster the customer into several clusters. Outliers can significantly skew the results of our data analysis and predictive modeling process. They can be caused by various factors such as measurement errors, data entry errors, or extreme variation in the data.
# - The outlier handling method that I use is capping the outliers to the lower/upper bound. This approach involves replacing the extreme values with a specified minimum and maximum value. It is a suitable method when we don't want to lose data, but at the same time, we want to limit the effect of the extreme values.

# %%
# Define a function to cap outliers in the data
def cap_outliers(data, columns):
    """
    
    Caps outliers in the specified columns of the dataframe.

    This function uses the IQR method to identify outliers. For each column specified,
    values below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR are considered outliers.
    These outliers are then capped at the lower and upper bounds respectively.

    Parameters:
    data (DataFrame): The dataframe to process.
    columns (list): The list of columns in wich to cap outliers.

    Returns:
    DatFrame: The processed dataframe with outliers capped.
    """
    
    # Create a copy of the data to avoid modifying the original dataframe
    result = data.copy()
    # Loop over each column specified
    for col in columns:
        # Calculate the first quartile (Q1)
        Q1 = result[col].quantile(0.25)
        # Calculate the third quartile (Q3)
        Q3 = result[col].quantile(0.75)
        # Calculate the Interquartile Range (IQR)
        IQR = Q3 - Q1
        # Define the lower bound as Q1 - 1.5 * IQR
        lower_bound = Q1 - 1.5 * IQR
        # Define the upper bound as Q3 + 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Replace values below the lower bound with the lower bound
        result[col] = np.where(result[col] < lower_bound, lower_bound, result[col])
        # Replace values above the upper bound with the upper bound
        result[col] = np.where(result[col] > upper_bound, upper_bound, result[col])
    # Return the dataframe with capped outliers
    return result

# Define the columns to cap outliers in
outliers = ['Income', 'MntMeatProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
            'NumWebPurchases', 'NumCatalogPurchases', 'NumWebVisitsMonth', 'Age', 'TotalTrx', 'ConversionRate'] 

# Apply the cap_outliers function to the dataframe 'dfp' and the specified columns
dfp_noutlier = cap_outliers(dfp, outliers)

# %% [markdown]
# #### Feature Selection
# - As said before, the `Z_CostContact` and `Z_Revenue` features only contain a single unique value. These features do not provide any variability or valuable information for our analysis or predictive modeling, and will be dropped in this section.
# - The `Unnamed: 0` feature appears to be an index column. Since Pandas DataFrames automatically provide an index, this column is redundant and will be dropped in this section.
# - The `Dt_Customer` feature is a date column. This feature will be dropped in this section because it is not needed in the machine learning model.
# - The `ID` feature is a unique identifier for each customer. This feature will be dropped in this section because it is not needed in the machine learning model.
# - The `Year_Birth` feature is a date column. This feature will be dropped in this section because it is not needed in the machine learning model.

# %%
# Drop unnecessary columns from the dataframe 'dfp_noutlier'
unnecessary_col = ['Unnamed: 0', 'ID', 'Year_Birth', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue']
dfp_noutlier = dfp_noutlier.drop(columns=unnecessary_col)
dfp_noutlier.sample(3)

# %% [markdown]
# #### Feature Encoding
# Features to label Encode :<br>
# - Education
# - Age Group
# because the features have an order of values(ordinal data)
# 
# Features to One Hot Encode: <br>
# - Marital_Status
# - Parent
# because the features don't have an order of values(nominal data)

# %%
# Label Encding
# Initialize Label Encoder as le
le = LabelEncoder()

dfp_noutlier['Education'] = le.fit_transform(dfp_noutlier['Education'])
dfp_noutlier['AgeGroup'] = le.fit_transform(dfp_noutlier['AgeGroup'])


# One hot Encoding
ms_encoded = pd.get_dummies(dfp_noutlier['Marital_Status'], prefix='Status').astype(int)
dfp_noutlier = pd.concat([dfp_noutlier, ms_encoded], axis=1)

parent_encoded = pd.get_dummies(dfp_noutlier['Parent'], prefix='Parent').astype(int)
dfp_noutlier = pd.concat([dfp_noutlier, parent_encoded], axis=1)

# drop marital status and parent column after encoded(redundant)
dfp_noutlier.drop(columns=['Marital_Status', 'Parent'], inplace=True)

print('\ndataframe after feature encoding :')
display(dfp_noutlier.head())

# %% [markdown]
# #### Standarization
# Standarization is needed to make the data have a mean of 0 and a standard deviation of 1. This is useful for Unsupervised Machine Learning models such as KMeans Clustering. Because the KMeans Clustering model uses the Euclidean Distance method to calculate the distance between data points, the data must be standardized so that the distance between data points is not too far apart.

# %%
# Inititalize standard scaler as scaler
scaler = StandardScaler()
# Standardize the data
scaled_data = scaler.fit_transform(dfp_noutlier)

# new dataframe with scaled data
scaled_dfp = pd.DataFrame(scaled_data, columns=dfp_noutlier.columns, index=dfp_noutlier.index)

print('\ndataframe after scaled(standarized) :')
scaled_dfp.head()

# %% [markdown]
# ### PCA
# PCA is needed to reduce the dimensionality of the data. Because the data has 39 features, it is necessary to reduce the dimensionality of the data so that the data is not too complex and the machine learning model can run faster.

# %%
# Initialize a PCA object with 2 components
# PCA (Principal Component Analysis) is a technique used to reduce the dimensionality of the data
pca = PCA(n_components=2, random_state=42)

# Fit the PCA model to the scaled data and transform the data into the first two principal components
# The transformed data is then converted into a DataFrame with the same index as 'dfp_noutlier'
dfpca = pd.DataFrame(pca.fit_transform(scaled_dfp), index=dfp_noutlier.index)

# Rename the columns of the DataFrame to 'PC1' and 'PC2' for better readability
# 'PC1' and 'PC2' represent the first and second principal components respectively
dfpca.rename(columns={0:'PC1', 1:'PC2'}, inplace=True)

# %% [markdown]
# ## Task 3 : Modelling
# Goals : Group customers into several clusters<br><br>
# Objective : 
# Apply the k-means clustering algorithm to the existing dataset, choose the correct number of clusters by looking at the elbow method, and evaluate using the silhouette score.

# %% [markdown]
# ### Find the optimal n cluster with Elbow Method and Silhouette Method 
# Elbow Method is used to find the optimal number of clusters by looking at the elbow point in the graph. The elbow point is the point where the graph starts to flatten.<br>
# And Silhouette Method is used to find the optimal number of clusters by looking at the highest silhouette score because the silhouette score is a metric used to calculate the distance between clusters. The higher the silhouette score, the better the cluster.

# %%
# Initialize empty lists to store inertia and silhouette scores for different numbers of clusters
inertia = []
silhouette = []

# Loop over a range of numbers from 2 to 9 (inclusive) to represent different numbers of clusters
for k in range(2, 10):
    # Initialize a KMeans object with 'k' clusters
    kmeans = KMeans(n_clusters=k, random_state=12, n_init=10)
    # Fit the KMeans model to the PCA-transformed data
    kmeans.fit(dfpca)
    # Append the inertia of the model to the 'inertia' list
    inertia.append(kmeans.inertia_)
    # Get the cluster labels predicted by the model
    cluster_label = kmeans.labels_
    # Calculate the silhouette score of the model and append it to the 'silhouette' list
    silhouette.append(silhouette_score(dfpca, cluster_label))

# Create a new figure and axis for plotting
fig, ax1 = plt.subplots()
fig.set_facecolor("#E8E8E8")

# Set the labels for the x-axis and the left y-axis
ax1.set_xlabel("k")
ax1.set_ylabel("inertia score", color="tab:blue")
# Plot the inertia scores against the number of clusters
ax1.plot(range(2, 10), inertia, marker="o", linestyle="--", color="tab:blue", label="inertia")
ax1.tick_params(axis="y", labelcolor="tab:blue")

# Create a second y-axis for the same x-axis
ax2 = ax1.twinx()
# Set the label for the right y-axis
ax2.set_ylabel("silhouette score", color="tab:red")
# Plot the silhouette scores against the number of clusters
ax2.plot(range(2, 10), silhouette, marker="o", linestyle="--", color="tab:red", label="silhouette")
ax2.tick_params(axis="y", labelcolor="tab:red")

# Combine the legends for the two plots
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc="upper right")

# Set the title for the plot and display the plot
plt.title("Inertia-Silhouette Score")
plt.show()

# %% [markdown]
# Based on the Elbow Method and Silhouette Method, the optimal number of clusters is 3 clusters, but I choose 4 clusters because the customer segmentation will be more detailed, and the silhouette score is still good.

# %%
# Create a 2x2 subplot with a specified figure size
fig, ax = plt.subplots(2, 2, figsize=(15, 8))
# Set the background color of the figure
fig.set_facecolor("#E8E8E8")

# Loop over a range of numbers from 2 to 5 (inclusive) to represent different numbers of clusters
for i in range(2, 6):
    # Initialize a KMeans object with 'i' clusters
    kmeans = KMeans(n_clusters=i, random_state=12, n_init=10)
    # Calculate the row (q) and column (mod) indices for the subplot
    q, mod = divmod(i, 2)
    # Initialize a SilhouetteVisualizer with the KMeans model and specify the colors to use
    visualizer = SilhouetteVisualizer(kmeans, colors="yellowbrick", ax=ax[q - 1][mod])
    # Fit the SilhouetteVisualizer to the PCA-transformed data
    visualizer.fit(dfpca)
    # Set the title for the subplot
    ax[q - 1][mod].set_title(f'Silhouette plot for {i} clusters', fontsize=12, fontweight='bold')
    # Set the x-label for the subplot
    ax[q - 1][mod].set_xlabel('Silhouette Coefficient Values')
    # Set the y-label for the subplot
    ax[q - 1][mod].set_ylabel('Cluster Label')
# Adjust the layout of the subplots to ensure that the plots do not overlap
plt.tight_layout()

# %% [markdown]
# From the second silhouette score analysis plot, it is evident that the data points within each of the four clusters are relatively close to each other, indicating a good degree of cohesion within each cluster. Furthermore, the average silhouette score remains consistent across the clusters, suggesting a balanced distribution of data points among them. This balance is a positive sign, as it indicates that no single cluster is overly dominant, which could potentially skew the overall analysis.

# %% [markdown]
# ### Fit KMeans Model
# In this section, I will fit the KMeans model with the optimal number of clusters, which is 4 clusters. The KMeans model will be fitted with the PCA data. and the result of the KMeans model will be saved in the `cluster` column.

# %%
# Set the optimal number of clusters to 4
k_optimal = 4

# Initialize a KMeans object with the optimal number of clusters
kmeans = KMeans(n_clusters=k_optimal, random_state=120, n_init='auto')

# Fit the KMeans model to the PCA-transformed data
kmeans.fit(dfpca)

# Add the cluster labels predicted by the model to the dataframe 'dfpca'
dfpca['Cluster'] = kmeans.labels_

# Display the dataframe 'dfpca'
dfpca

# %%
# Create a new figure with a specified size and background color
plt.figure(figsize=(12,8), facecolor='#E8E8E8')

# Create a scatter plot of the first principal component ('PC1') against the second principal component ('PC2')
# The points are colored according to their cluster label
palt = ['#D1106F', '#00D19B', '#25A9D9', '#D16F11']
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=dfpca, palette=palt)

# Get the coordinates of the cluster centers from the KMeans model
centroids = kmeans.cluster_centers_
# Plot the cluster centers as black 'x' markers
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.8, marker='x')

# Set the title for the plot
plt.title('K-Means Clustering', fontsize=18, fontweight='bold', y=1.03)
# Set the label for the x-axis
plt.xlabel('PCA 1', fontsize=12)
# Set the label for the y-axis
plt.ylabel('PCA 2', fontsize=12)
# Remove the grid from the plot
plt.grid(False)
# Display the plot
plt.show()

# %% [markdown]
# Based on the KMeans model, the data points are divided into 4 clusters, namely cluster 0, cluster 1, cluster 2, and cluster 3. The data points in each cluster are not too far apart, so the KMeans model is good enough to cluster the data points.

# %% [markdown]
# ## Task 4 - Customer Personality Analysis For Marketing Retargeting
# **Goal** : Develop a targeted marketing strategy to optimize costs and boost revenue.<br><br>
# **Objective** : 
# - Interpretate the clusters
# - Use visualizations to identify key characteristics and behaviors of each customer group. This will help us understand our customers better and tailor our marketing strategies to their specific needs and preferences.

# %%
# Create a copy of the dataframe 'dfp_noutlier' and store it in 'df_clust'
df_clust = dfp_noutlier.copy()

# Get the 'cluster' column from the dataframe 'dfpca'
label = dfpca['Cluster']

# Add the 'cluster' column to the dataframe 'df_clust'
df_clust['Cluster'] = label

df_clust

# %%
# Define a list of metrics for creating box plots
metrics = ['Income', 'TotalSpending', 'ConversionRate', 'TotalTrx', 'Cluster']
# Remove 'cluster' from the list as it is not a metric
metrics.remove('Cluster')

# Calculate the number of rows needed for the subplots
n = len(metrics)
ncols = 2
nrows = 2  # Set the number of rows to 2 for a 2x2 grid

# Create a figure and a grid of subplots
fig, ax = plt.subplots(nrows, ncols, figsize=(20, nrows*5))
fig.set_facecolor('#E8E8E8')  # Set the figure background color

# Flatten the axes array for easier iteration
ax = ax.flatten()

# Define the order of clusters for display in the box plots
cluster_order = [2,0,3,1]

# Define the color palette
palt = ['#D1106F', '#00D19B', '#25A9D9', '#D16F11']

# Create subplots for each feature
for i, feature in enumerate(metrics):
    # Create a box plot for the current feature
    sns.boxplot(data=df_clust, y=feature, x='Cluster', hue='Cluster', palette=palt, ax=ax[i], order=cluster_order, hue_order=cluster_order, linewidth=2.1, width=0.65)
    ax[i].set_title(feature)  # Set the title of the subplot
    ax[i].grid(False)  # Remove the grid from the subplot
    ax[i].legend(loc='center left', bbox_to_anchor=(1, 0.8))  # Move the legend to the outside of the subplot
    # Change the labels of the hue legend to more descriptive labels
    hue_labels = ['High Spender', 'Mid Spender', 'Low Spender', 'Risk Churn']
    legend = ax[i].get_legend()
    for text, label in zip(legend.texts, hue_labels):
        text.set_text(label)

# Adjust the layout of the subplots to prevent overlapping display the figure
plt.tight_layout()  
plt.show()  

# %%
# Create a copy of the dataframe 'df_clust' and store it in 'interpretation'
interpretation = df_clust.copy()

# Rename cluster value to more descriptive labels
interpretation['Cluster'] = interpretation['Cluster'].map({
    2: 'High Spender', 
    0: 'Mid Spender', 
    3: 'Low Spender', 
    1: 'Risk Churn'
})

# Rename the columns to more descriptive labels
rename_cols = {
    'MntCoke': 'CokeProducts',
    'MntFruits': 'FruitsProducts',
    'MntMeatProducts': 'MeatProducts',
    'MntFishProducts': 'FishProducts',
    'MntSweetProducts': 'SweetProducts',
    'MntGoldProds': 'GoldProducts'
}
interpretation.rename(columns=rename_cols, inplace=True)
# Define the columns to be used in the groupby operation
intr_metrics = [
    'ConversionRate',
    'TotalSpending', 
    'Income',
    'TotalTrx',
    'TotalAcceptedCmp',
    'CokeProducts', 
    'FruitsProducts', 
    'MeatProducts', 
    'FishProducts', 
    'SweetProducts', 
    'GoldProducts'
]
# Set the float format to display numbers with a thousands separator
pd.options.display.float_format = '{:,.2f}'.format

# Calculate and display the total sum of each metric for each cluster
sum_interpretation = interpretation.groupby('Cluster')[intr_metrics].sum().sort_values('TotalSpending', ascending=False).astype(float).reset_index()
print('Total/SUM Clusters Metrics :\n')
display(sum_interpretation)

# Calculate and display the average value of each metric for each cluster
average_interpretation = interpretation.groupby('Cluster')[intr_metrics].mean().sort_values('TotalSpending', ascending=False).astype(float).reset_index()
print('\n\nAverage/Mean Clusters Metrics :\n')
display(average_interpretation)

# Calculate and display the maximum value of each metric for each cluster
max_interpretation = interpretation.groupby('Cluster')[intr_metrics].max().sort_values('TotalSpending', ascending=False).astype(float).reset_index()
print('\n\nMax Clusters Metrics :\n')
display(max_interpretation)

# %%
fig, ax = plt.subplots(2, 5, figsize=(30, 15))
fig.set_facecolor('#E8E8E8')

palt = ['#D1106F', '#00D19B', '#25A9D9', '#D16F11']
orderseg = ['High Spender', 'Mid Spender', 'Low Spender', 'Risk Churn']

# loop over each feature and create a bar plot on a separate subplot
for feature, ax in zip(intr_metrics, ax.flatten()):
    # Create a bar plot for the current feature
    barplot = sns.barplot(data=sum_interpretation, x='Cluster', y=feature, ax=ax, palette=palt, hue='Cluster', order=orderseg, hue_order=orderseg, errorbar=None, edgecolor='black')
    ax.set_title(f'SUM {feature} by Cluster', fontsize=15, fontweight='bold', pad=5)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel(feature, fontsize=12)
    ax.grid(False)

    # Add number annotations inside the bars
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.2f'), 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha = 'center', va = 'center', 
                         xytext = (0, 12), 
                         textcoords = 'offset points')

plt.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(2, 5, figsize=(30, 15))
fig.set_facecolor('#E8E8E8')

# loop over each feature and create a bar plot on a separate subplot
for feature, ax in zip(intr_metrics, ax.flatten()):
    # Create a bar plot for the current feature
    barplot = sns.barplot(data=average_interpretation, x='Cluster', y=feature, ax=ax, palette=palt, hue='Cluster', order=orderseg, hue_order=orderseg, errorbar=None, edgecolor='black')
    ax.set_title(f'AVERAGE {feature} by Cluster', fontsize=15, fontweight='bold', pad=10)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel(feature, fontsize=12)
    ax.grid(False)

    # Add number annotations inside the bars
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.2f'), 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha = 'center', va = 'center', 
                         xytext = (0, 12), 
                         textcoords = 'offset points')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Cluster Interpretation and Marketing Recommendations:
# 
# #### 1. **High Spender:**
#    - **Cluster Characteristics:**
#      - Highest Sum Total Spending: $705,148,000.00
#      - Highest Average Conversion Rate: 2.13
#      - Highest Average Income: $74,856,542.64
#      - Average Total Transaction : 20.13
#      - Top Spending Categories: Coke and Meat Products
# 
#    - **Interpretation:**
#      - This cluster comprises customers who exhibit an exceptional conversion rate, spending the most on the platform.
#      - Their high income and infrequent visits indicate that when they engage, they make substantial purchases.
#      - The primary spending focus is on Coke and meat products.
# 
#    - **Marketing Recommendation:**
#      - Implement personalized retargeting campaigns based on previous purchases, emphasizing exclusive offers on Coke and meat products.
#      - Utilize high-income targeting for promotions and loyalty programs to further increase customer spend.
# 
# #### 2. **Mid Spender:**
#    - **Cluster Characteristics:**
#      - Sum Total Spending: $411,483,000.00
#      - Average Conversion Rate: 1.44
#      - Average Income: $65,118,012.41
#      - Highest Average Total Transaction 23.58
#      - Top Spending Categories: Coke and Meat Products
# 
#    - **Interpretation:**
#      - This group represents customers with a good conversion rate, showing a moderate level of spending and income.
#      - The Average Total Transaction this cluster has is higher than the high spender.   
#      - Similar to the High Spender cluster, Coke and meat products are significant areas of expenditure.
# 
#    - **Marketing Recommendation:**
#      - Implement retargeting strategies to promote a broader range of products to increase the average transaction value.
#      - Consider introducing loyalty programs to encourage more frequent visits and higher spending.
# 
# #### 3. **Low Spender:**
#    - **Cluster Characteristics:**
#      - Sum Total Spending: $160,035,000.00
#      - Average Conversion Rate: 0.84
#      - Average Income: $50,110,625.90
#      - Average Total Transaction 16.40
#      - Top Spending Categories: Coke and Meat Products
# 
#    - **Interpretation:**
#      - This cluster represents customers with lower conversion rates and spending levels compared to the previous clusters.
#      - While income is relatively high, the spending behavior suggests potential for increased engagement.
# 
#    - **Marketing Recommendation:**
#      - Implement targeted promotions for a wider range of products to encourage increased spending.
#      - Launch special offers and discounts to attract this segment and increase their frequency of visits.
# 
# #### 4. **Risk Churn:**
#    - **Cluster Characteristics:**
#      - Lowest Sum Total Spending: $68,981,000.00
#      - Lowest Average Conversion Rate: 0.30
#      - Average Income: $32,992,111.42	
#      - Average Total Transaction 7.12
#      - Top Spending Categories: Coke and Meat Products
# 
#    - **Interpretation:**
#      - This cluster represents customers with a relatively low conversion rate, lower spending, and potential risk of churn.
#      - The lowest average conversion rate means this cluster often visits our online web platform but doesn't finish the transaction.
#      - Similar spending patterns in Coke and meat products, but the lower conversion rate indicates a need for targeted retention efforts.
# 
#    - **Marketing Recommendation:**
#      - Implement aggressive retargeting campaigns with personalized incentives to prevent churn.
#      - Focus on customer satisfaction initiatives and exclusive offers to re-engage and retain customers.
# 
# ### General Recommendations:
# - **Cross-Sell Strategies:**
#   - Leverage data to identify cross-sell opportunities within each cluster, encouraging customers to explore additional product categories.
# 
# - **Dynamic Pricing:**
#   - Implement dynamic pricing strategies based on customer behavior and purchase history to optimize revenue from each cluster.
# 
# - **Customer Segmentation Refinement:**
#   - Regularly review and refine customer segmentation to adapt to changing market dynamics and customer preferences.
# 
# - **Invest in Data Analytics:**
#   - Continue investing in advanced analytics to uncover deeper insights and refine marketing strategies based on evolving customer behaviors.


