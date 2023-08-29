#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required libraries and dependencies
import pandas as pd
import hvplot.pandas
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[2]:


# Load the data into a Pandas DataFrame
df_market_data = pd.read_csv(
    "crypto_market_data.csv",
    index_col="coin_id")

# Display sample data
df_market_data.head(10)


# In[3]:


# Generate summary statistics
df_market_data.describe()


# In[4]:


# Plot your data to see what's in your DataFrame
df_market_data.hvplot.line(
    width=800,
    height=400,
    rot=90
)


# ---

# ### Prepare the Data

# In[5]:


# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
scaled = StandardScaler().fit_transform(df_market_data[["price_change_percentage_24h", "price_change_percentage_7d",
        "price_change_percentage_14d", "price_change_percentage_30d",
        "price_change_percentage_60d", "price_change_percentage_200d",
        "price_change_percentage_1y"]])


# In[6]:


# Create a DataFrame with the scaled data
df_transformed = pd.DataFrame(scaled, columns=["price_change_percentage_24h", "price_change_percentage_7d",
        "price_change_percentage_14d", "price_change_percentage_30d",
        "price_change_percentage_60d", "price_change_percentage_200d",
        "price_change_percentage_1y"])

# Copy the crypto names from the original data
df_transformed["coin_id"] = df_market_data.index
# Set the coinid column as index
df_transformed = df_transformed.set_index("coin_id")


# Display sample data
df_transformed.head()


# ---

# ### Find the Best Value for k Using the Original Data.

# In[7]:


# Create a list with the number of k-values from 1 to 11
k_val=list(range(1,11))


# In[9]:


# Create an empty list to store the inertia values
inertia_values = []


# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_scaled`
# 3. Append the model.inertia_ to the inertia list
for i in k_val:
    k_model = KMeans(n_clusters=i)
    k_model.fit(df_transformed)
    inertia_values.append(k_model.inertia_)


# In[11]:


# Create a dictionary with the data to plot the Elbow curve
Elb_curve={"k": k_val,"inertia": inertia_values}

# Create a DataFrame with the data to plot the Elbow curve
dfelbows=pd.DataFrame(Elb_curve)
dfelbows.head


# In[12]:


# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
dfelbows.hvplot.line(x="k", y="inertia", xticks=k_val)


# #### Answer the following question: 
# 
# **Question:** What is the best value for `k`?
# 
# **Answer:** 4

# ---

# ### Cluster Cryptocurrencies with K-means Using the Original Data

# In[13]:


# Initialize the K-Means model using the best value for k

model = KMeans(n_clusters=4)


# In[16]:


# Fit the K-Means model using the scaled data

model.fit(df_transformed)


# In[19]:


# Predict the clusters to group the cryptocurrencies using the scaled data

predict_clusters=model.predict(df_transformed)

# Print the resulting array of cluster values.

predict_clusters


# In[20]:


# Create a copy of the DataFrame
df_copy=df_transformed.copy()


# In[21]:


# Add a new column to the DataFrame with the predicted clusters
df_copy["predicted_clusters"] = predict_clusters


# Display sample data
df_copy.head()


# In[22]:


# Create a scatter plot using hvPlot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
df_copy.hvplot.scatter(x="price_change_percentage_24h", y= "price_change_percentage_7d", by="predicted_clusters", hover_cols= "coin_id")


# ---

# ### Optimize Clusters with Principal Component Analysis.

# In[23]:


# Create a PCA model instance and set `n_components=3`.
pca= PCA(n_components=3)


# In[25]:


# Use the PCA model with `fit_transform` to reduce to 
# three principal components.
fit_t=pca.fit_transform(df_transformed)
# View the first five rows of the DataFrame. 
fit_t[:5]


# In[26]:


# Retrieve the explained variance to determine how much information 
# can be attributed to each principal component.
pca.explained_variance_ratio_



# In[27]:


sum(pca.explained_variance_ratio_)


# #### Answer the following question: 
# 
# **Question:** What is the total explained variance of the three principal components?
# 
# **Answer:** ABout 90%

# In[38]:


# Create a new DataFrame with the PCA data.
pca_df=pd.DataFrame(fit_t, columns=["pc1","pc2","pc3"])
# Creating a DataFrame with the PCA data

# Copy the crypto names from the original data


# Set the coinid column as index
pca_df["coin_id"] = pca_df.index


pca_df = pca_df.set_index("coin_id")
# Display sample data
pca_df.head()


# ---

# ### Find the Best Value for k Using the PCA Data

# In[39]:


# Create a list with the number of k-values from 1 to 11
k_val=list(range(1,11))


# In[40]:


# Create an empty list to store the inertia values
inertia_values = []


# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_pca`
# 3. Append the model.inertia_ to the inertia list
for i in k_val:
    k_model = KMeans(n_clusters=i)
    k_model.fit(pca_df)
    inertia_values.append(k_model.inertia_)


# In[42]:


# Create a dictionary with the data to plot the Elbow curve
pcaElb_curve={"k": k_val,"inertia": inertia_values}
# Create a DataFrame with the data to plot the Elbow curve
elb_curve_data=pd.DataFrame(pcaElb_curve)
elb_curve_data.head()


# In[43]:


# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
elb_curve_data.hvplot.line( x= "k", y ="inertia")


# #### Answer the following questions: 
# 
# * **Question:** What is the best value for `k` when using the PCA data?
# 
#   * **Answer:** 4
# 
# 
# * **Question:** Does it differ from the best k value found using the original data?
# 
#   * **Answer:** No

# ### Cluster Cryptocurrencies with K-means Using the PCA Data

# In[44]:


# Initialize the K-Means model using the best value for k
model=KMeans(n_clusters=4)


# In[45]:


# Fit the K-Means model using the PCA data
model.fit(pca_df)


# In[46]:


# Predict the clusters to group the cryptocurrencies using the PCA data
predict_clus_gr=model.predict(pca_df)

# Print the resulting array of cluster values.
print(predict_clus_gr)


# In[47]:


# Create a copy of the DataFrame with the PCA data
copy_pcadf= pca_df.copy()

# Add a new column to the DataFrame with the predicted clusters
copy_pcadf["predicted_clusters"] = predict_clus_gr

# Display sample data
copy_pcadf


# In[48]:


# Create a scatter plot using hvPlot by setting 
# `x="PC1"` and `y="PC2"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
copy_pcadf.hvplot.scatter(x="pc1", y= "pc2", by="predicted_clusters", hover_cols="coin_id") 


# ### Visualize and Compare the Results
# 
# In this section, you will visually analyze the cluster analysis results by contrasting the outcome with and without using the optimization techniques.

# In[51]:


# Composite plot to contrast the Elbow curves
dfelbows.hvplot.line(x="k", y="inertia", xticks=k_val) + elb_curve_data.hvplot.line( x= "k", y ="inertia")


# In[50]:


# Composite plot to contrast the clusters
df_copy.hvplot.scatter(x="price_change_percentage_24h", y= "price_change_percentage_7d", by="predicted_clusters", hover_cols= "coin_id") + copy_pcadf.hvplot.scatter(x="pc1", y= "pc2", by="predicted_clusters", hover_cols="coin_id") 


# #### Answer the following question: 
# 
#   * **Question:** After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?
# 
#   * **Answer:** 
