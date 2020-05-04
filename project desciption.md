## Project Overview

In my capstone project, I aim to answer 2 main business questions:
1. What are the main drivers of an effective offer on the Starbucks app?
2. Could the data provided, namely offer characteristics and user demographics, predict whether a user would take up an offer?

This capstone project is using data provided by Udacity as part of the Data Scientist Nanodegree course. It contains simulated data that mimics customer behavior on the Starbucks rewards mobile app.

The provided background information on the mobile app is that once every few days, Starbucks sends out an offer to users of the mobile app. Some users might not receive any offer during certain weeks, and not all users receive the same offer.

## Problem Statement

As stated above, the problem statement I am aiming to answer are to (1) discover the main drivers of offer effectiveness, and (2) explore if we can predict whether a user would take up an offer.

The data provided consists of 3 datasets:
- Offer portfolio, which consists of the attributes of each offer
- Demographic data for each customer
- Transactional records of events occurring on the app

Using the data provided, I answer the above two questions using 3 classification supervised machine learning models, feeding in the data from three different offer types.

I use the model to uncover the feature importances to identify the drivers of offer effectiveness, while exploring if the model itself could be used to predict if a user would take up an offer.

Lastly, I also explore the characteristics of users who do or do not take up an offer.

My project aims to answer the two questions above, but I also ended up adding 2 additional models as points of exploration - the first assessing whether an all-in-one model could be used in place of 3 different models, with the offer types functioning as a categorical variable. Secondly, I also build a regression model to see if we could predict the amount a user would spend, given that the offer is effectively influencing them.
