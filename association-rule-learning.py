#########################
# Business Problem
#########################

# Armut, Turkey's largest online service platform, connects service providers with those who want to receive services.
# It enables easy access to services such as cleaning, renovation, and transportation with just a few clicks
# from a computer or smartphone.
# Using the dataset that contains service-receiving users and the services and categories they have received,
# it is aimed to build a product recommendation system using Association Rule Learning.


#########################
# Dataset
#########################
# The dataset consists of the services received by customers and their categories.
# It contains the date and time information for each service taken.

# UserId: Customer ID
# ServiceId: Anonymized services belonging to each category (e.g., sofa cleaning under the cleaning category)
# A ServiceId can be found under different categories and represent different services.
# (e.g., the service with CategoryId 7 and ServiceId 4 refers to radiator cleaning, while the service with CategoryId 2 and ServiceId 4 refers to furniture assembly)
# CategoryId: Anonymized categories (e.g., Cleaning, Transportation, Renovation)
# CreateDate: The date when the service was purchased

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


#########################
# TASK 1: Data Preparation
#########################

# Step 1: Read the "armut_data.csv" file.

df_ = pd.read_csv("armut_data.csv")
df = df_.copy()
df.info()

# Step 2: The ServiceID represents a different service for each CategoryID.
# Create a new variable representing services by combining ServiceID and CategoryID with an underscore "_".

df["Service"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)

df.head()

# Step 3: The dataset consists of the date and time when services were received, without any basket (invoice, etc.) definition.
# To apply Association Rule Learning, a basket definition must be created.
# Here, the basket definition is the monthly services received by each customer.
# For example, the services 9_4 and 46_4 received by the customer with ID 7256 in August 2017 constitute one basket;
# while services 9_4 and 38_4 received in October 2017 form another basket. Each basket needs to be defined with a unique ID.
# First, create a new "date" variable containing only the year and month. Then, combine UserID and the new "date" variable
# using an underscore "_" and assign it to a new variable named "ID".

df["date"] = pd.to_datetime(df["CreateDate"]).dt.strftime('%Y-%m')
df["basketID"] = df["UserId"].astype(str) + "_" + df["date"].astype(str)

df.head(20)

#########################
# TASK 2: Generate Association Rules
#########################

# Step 1: Create a basket-service pivot table as shown below.

# Service         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# basketID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..

service_usage_df = df.groupby(['basketID', 'Service']).agg({"Service": "count"}).unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)


# Step 2: Create association rules.

from mlxtend.frequent_patterns import apriori, association_rules
service_usage_itemsets = apriori(service_usage_df, min_support=0.01, use_colnames=True)

service_usage_itemsets.sort_values("support", ascending=False)

service_usage_rules = association_rules(service_usage_itemsets, metric="lift", min_threshold=0.01)

sorted_rules = service_usage_rules.sort_values("lift", ascending=False)


# Step 3: Use the arl_recommender function to suggest a service to a user who recently received the 2_0 service.

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j[1] == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count]

arl_recommender(service_usage_rules, "2_0", 3)
