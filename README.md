<p align="center">
  <a href="#">
    <img src="https://badges.pufler.dev/visits/azaryasph/minpro-predict-customer-personality" alt="Visits Badge">
    <img src="https://badges.pufler.dev/updated/azaryasph/minpro-predict-customer-personality" alt="Updated Badge">
    <img src="https://badges.pufler.dev/created/azaryasph/minpro-predict-customer-personality" alt="Created Badge">
    <img src="https://img.shields.io/github/contributors/azaryasph/minpro-predict-customer-personality" alt="Contributors Badge">
    <img src="https://img.shields.io/github/last-commit/azaryasph/minpro-predict-customer-personality" alt="Last Commit Badge">
    <img src="https://img.shields.io/github/commit-activity/m/azaryasph/minpro-predict-customer-personality" alt="Commit Activity Badge">
    <img src="https://img.shields.io/github/repo-size/azaryasph/minpro-predict-customer-personality" alt="Repo Size Badge">
    <img src="https://img.shields.io/badge/contributions-welcome-orange.svg" alt="Contributions welcome">
    <img src="https://www.codefactor.io/repository/github/azaryasph/minpro-predict-customer-personality" alt="CodeFactor" />
  </a>
</p>

# <img src="https://yt3.googleusercontent.com/ytc/AIdro_n0EO16H6Cu5os9ZOOP1_BsmeruHYlmlOcToYvI=s900-c-k-c0x00ffffff-no-rj" width="30"> Mini Project 3: Predict Customer Personality to Boost Marketing Campaign using Machine Learning <img src="https://yt3.googleusercontent.com/ytc/AIdro_n0EO16H6Cu5os9ZOOP1_BsmeruHYlmlOcToYvI=s900-c-k-c0x00ffffff-no-rj" width="30">

<p align="center">
    <img src="https://images.unsplash.com/photo-1475275083424-b4ff81625b60?q=80&w=2072&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="Hotel" width="600" height="320">
</p>

Photo by <a href="https://unsplash.com/@cbyoung?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Clark Young</a> on <a href="https://unsplash.com/photos/yellow-shopping-carts-on-concrete-ground-ueZXMrZFFKQ?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a>

## Table of Contents 📚
1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Conversion Rate Analysis](#conversion-rate-analysis)
7. [Data Preprocessing](#data-preprocessing)
8. [Modeling](#modeling)
9. [Customer Personality Analysis For Marketing Retargeting](#customer-personality-analysis-for-marketing-retargeting)
10. [Conclusion](#conclusion)
11. [Acknowledgements](#acknowledgements)

## Introduction 📢
This project is provided by Rakamin Academy as a mini project for the Data Science Bootcamp Job Acceleration Program, The project is about predicting customer personality in a online market platform to boost the marketing campaign using machine learning. The dataset used in this project is also provided by Rakamin Academy.

## Project Overview 📋
A company can develop rapidly when it knows its customers' personality behaviour to provide better services and benefits to customers who can become loyal customers. By processing historical marketing campaign data to imporve performance and target the right customers so they can make transactions on the company's platform, from this data insight, our focus is to create cluster prediction model to make it easier for companies to make decisions in the future.

## Requirements 📦
This project requires the following packages:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `yellowbrick`

## Installation 🔁
To install the required packages, you can run the following command:
```bash
pip install -r requirements.txt
```

## Usage 🪴
To run the Jupyter Notebook, follow these steps:
1. Clone this repository to your local machine
2. Open the terminal and navigate to the project directory
3. Run the following command:
```bash
jupyter notebook
```
4. The Jupyter Notebook will open in your browser
5. Click on the `Mini Project 3 - Predict Customer Personality to Boost Marketing Campaign using Machine Learning.ipynb` file to open the notebook

## Conversion Rate Analysis 📊
In this section, we will analyze the conversion rate of the marketing campaign data.
But before we analyze the conversion rate, we need to create a new features :
1. `AgeGroup`
2. `TotalSpending`
3. `Parent`
4. `NumChild`
5. `ConversionRate` 
6. `TotalSpending`
7. `TotalTrx`
8. `TotalAcceptedCampaign`

After creating the new features, Let's analyze the conversion rate of the marketing campaign data.

### Age Group, Parent Status, Education Level, and Marital Status Distribution 🧮

<p float="left">
  <img src="./src\images/image-4.png" width="550" />
  <img src="./src\images/image-5.png" width="600" /> 
</p>

Based on our analysis of the customer data, we have identified several important findings regarding the demographic profile of our primary customer base:

- Age Group: Our customer base primarily consists of individuals in the middle-aged group, particularly those between the ages of 40 and 59. It seems that our products or services are quite popular among people in this age range.

- Family Situation: A considerable number of our customers have children living with them. Our offerings may cater well to the needs of parents or families, indicating a potential fit.

- Education Level: The majority of our customers have completed at least a Bachelor's Degree. This could indicate the affordability, accessibility, or attractiveness of our products or services to individuals with this level of education.

- Relationship Status: Most of our customers are married. This suggests that our products or services are favoured by couples or tailored to the needs of married individuals.

These insights can provide valuable information for gaining a deeper understanding of our customer base and customising our marketing strategies, product development, and services to align with their specific needs and preferences. Nevertheless, it's crucial to keep in mind that these are overall patterns and there might be notable differences within these categories. Additional segmentation and analysis may offer more detailed insights.

### Total Spending, Income, and Conversion Rate Correlation 🛒

<p align="center">
  <img src="./src\images/image-6.png" width="550" />
</p>

Several significant correlations between the variables were noted in this multivariate analysis:

- **Amount Received and Conversion Rate**:Income and conversion rate have a positive relationship. This implies that customers are more likely to finish a purchase on our web platform after visiting if their income rises. This might be the result of having more discretionary income, which gives one greater freedom to choose what to buy.

- **Total Expenditure and Revenue**:The relationship between total spending and income is positive. This suggests that consumers tend to spend more when their salaries are higher. This may indicate that they have more purchasing power.

- **The Total Amount Spent and the Conversion Rate**: The relationship between total spending and conversion rate is positive. This implies that after browsing our web platform, buyers who spend more are also more likely to finish a purchase. This can be the result of increased interest or engagement with our offerings.

- **Age and Conversion Rate**: Age and conversion rate do not significantly correlate. This suggests that there is no discernible age-related difference in the likelihood that a client will complete a transaction after browsing our web platform. This may indicate that a broad spectrum of age groups find our platform appealing.

These observations can improve our comprehension of consumer behavior and guide our approach to marketing and sales. It's crucial to keep in mind nevertheless that correlation does not imply causation, and more research may be required to determine the underlying reasons of these associations.

### Conversion Rate based on Age Group, Parent Status, Education Level, Num Child & Education Level 📈

<p align="center">
  <img src="./src\images/image-7.png" width="550" />
</p>

Through my analysis of customer behavior, I have discovered several significant trends:
- Customers in the Old Adults age group (> 59 years) have the highest conversion rate. Additionally, this age group also has the most significant spending, with a total expenditure of over 700,000. This suggests a strong level of trust and engagement in the online shopping experience. They might have a higher chance of making a purchase since they have a good grasp of the process and feel at ease with it.

- Customers without children or parenting experience tend to have a higher conversion rate compared to customers who already have children. On the other hand, the pie chart depicting our customer distribution reveals that a majority of our customers are either married or have children. Thus, customers without children have a greater chance of achieving a higher conversion rate.

- Customers without a degree (still in high school) exhibit the lowest conversion rate in comparison to customers who have already obtained a degree.


## Data Preprocessing 🛠️
In this section, we will preprocess the data by:

- Handling missing values (fill missing values with mean or median)

- Handle outliers (Cap the outliers)

- Feature selection (drop unnecessary features)

- Feature encoding (label encode for ordinal data & one hot encode for nominal data)

- Standarize the data using StandardScaler

- Perform PCA to reduce the dimensionality of the data into 2 components

## Modeling 🤖
In this section, we will build a clustering model using KMeans algorithm to predict customer personality,
before we build the model, we need to determine the optimal number of clusters using the Elbow Method and Silhouette Score.

<p float="left">
  <img src="./src\images/image.png" width="500" />
  <img src="./src\images/image-1.png" width="627" /> 
</p>

Based on the Elbow Method and Silhouette Score, we will use 4 clusters to build the model.

After determining the optimal number of clusters, we will build the KMeans model and visualize the clusters using PCA.

<p align="center">
    <img src="./src\images/image-2.png" width="500" />
</p>

Seems like the clusters are well separated, we can use this model to predict customer personality.

## Customer Personality Analysis For Marketing Retargeting 🎯
In this section, we interpret the clusters to understand the customer personality and provide insights for marketing retargeting.

<p align="center">
  <img src="./src\images/image-3.png" width="700" />
</p>

Based on the analysis, we can provide insights for marketing retargeting:

### 1. **High Spender:**

   - **Interpretation:**
     - This cluster comprises customers who exhibit an exceptional conversion rate, spending the most on the platform.
     - Their high income and infrequent visits indicate that when they engage, they make substantial purchases.
     - The primary spending focus is on Coke and meat products.

   - **Marketing Recommendation:**
     - Offer Exclusive Products<br>
     Offer a limited/premium version of an existing product that matches the purchase preferences of this previous customer cluster, namely Meat and Beverages. This can include larger quantities, even high quality, and unique packaging.

      - Subscription Services <br>
      Offers subscription services for the category of their choice (for example, monthly meat delivery boxs). It can provide convenience and recurring income.

      - Loyalty Programs<br>
      Implement a ranked loyalty program in return for significant benefits for customers who shop a lot. This can include personalized offers, early access to sales, or exclusive customer service.

### 2. **Mid Spender:**
  
   - **Interpretation:**
     - This group represents customers with a good conversion rate, showing a moderate level of spending and income.
     - The Average Total Transaction this cluster has is higher than the high spender.   
     - Similar to the High Spender cluster, Coke and meat products are significant areas of expenditure.

   - **Marketing Recommendation:**
     - Bundle deals and promotions<br>
     Provide product packages or discounts that align with their purchase preferences to increase their basket size and overall spending.
     
     - Free samples or trial subscription<br>
     Offer complimentary samples or trial subscriptions for new products that align with their purchase preferences to entice them to explore new options and boost their spending.
     
     - Highlight value proporsitons<br>
     Effectively convey the numerous advantages of shopping on our platform by highlighting competitive prices, high-quality products, and convenient shipping options.

### 3. **Low Spender:**
 
   - **Interpretation:**
     - This cluster represents customers with lower conversion rates and spending levels compared to the previous clusters.
     - While income is relatively high, the spending behavior suggests potential for increased engagement.

   - **Marketing Recommendation:**
     - Targeted Discount Offers and Flash Sales<br>
     Leverage their purchase history and favorite categories (meat and cola) to offer targeted discounts and flash sales. Emphasize affordability by highlighting percentage discounts or specific dollar-amount savings.
     
     - Value-Added Bundles and Upsells<br>
     Create value-added bundles that combine discounted staples (like meat and cola) with complementary products at a slight price increase. This can encourage them to spend slightly more while perceiving greater value.
     
     - Free Shipping Thresholds and Reward Programs<br>
     Implement a free shipping threshold specifically tailored to the average purchase amount of low spenders. This can incentivize them to increase their basket size to reach the free shipping threshold.

### 4. **Risk Churn:**

   - **Interpretation:**
     - This cluster represents customers with a relatively low conversion rate, lower spending, and potential risk of churn.
     - The lowest average conversion rate means this cluster often visits our online web platform but doesn't finish the transaction.
     - Similar spending patterns in Coke and meat products, but the lower conversion rate indicates a need for targeted retention efforts.

   - **Marketing Recommendation:**
     - Win-Back Campaign<br>
     Identify customers who last bought a long time ago and submit targeted win-back campaigns that offer personalized discounts, exclusive offers, or free shipping to push them back.
     
     - Feedback and Surveys<br>
     Conduct a survey or income poll to gather direct input from this segment about their experience, preferences, and reasons for their potential churn. Use this input to enhance our offer and address existing issues.
     
     - Loyalty Programs Tiers<br>
     Implement the loyalty program as previously recommended, but offer exclusive rewards or higher discounts on this segment to encourage repeat purchases and build loyalty.


### General Recommendations:
- **Cross-Sell Strategies:**
  - Leverage data to identify cross-sell opportunities within each cluster, encouraging customers to explore additional product categories.

- **Dynamic Pricing:**
  - Implement dynamic pricing strategies based on customer behavior and purchase history to optimize revenue from each cluster.

- **Customer Segmentation Refinement:**
  - Regularly review and refine customer segmentation to adapt to changing market dynamics and customer preferences.

- **Invest in Data Analytics:**
  - Continue investing in advanced analytics to uncover deeper insights and refine marketing strategies based on evolving customer behaviors.

## Conclusion 📝
In this project, we have successfully analyzed the conversion rate of the marketing campaign data, preprocessed the data, built a clustering model using KMeans algorithm to predict customer personality, and provided insights for marketing retargeting based on the clusters.

## Acknowledgements 🌟
I would like to express my gratitude to [Rakamin Academy](https://www.rakamin.com/) for providing the dataset and the opportunity to work on this project. I would also like to thank [Mr. Rezki Trianto](https://www.linkedin.com/in/rezkitrianto/) for the guidance and support throughout the project.

<p align="center">
    <img src="./src\images/Thankyou.jpg" alt="swag" width=600>
</p>


