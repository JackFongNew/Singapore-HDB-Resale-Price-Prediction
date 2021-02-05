# Capstone Project: Singapore HDB Flat Resale Price Prediction

## Introduction

In 1960, there was a housing crisis in Singapore. Many people were living in unhygienic slums and crowded squatter settlements. Only 9% of Singaporeans lived in government flats, while others yearned for a place to call home.
The Housing & Development Board (HDB) was thus set up on 1 February 1960, tasked to solve Singapore’s housing crisis by building the now well known “HDB Flat”.

In less than 3 years, it had built 21,000 HDB flats; 2 years later, that number was 54,000. Within a brief span of 10 years, HDB built a sufficient number of flats for Singaporeans and resolved the housing crisis.

Today Singapore’s HDB programme is recognised as one of the best public housing programmes globally. 

This affordable programme contributes significantly to the comfort of Singapore lifestyle as accommodation expenses can be a significant challenge in many counties around the world. This is particularly welcomed today as Singapore has been identified by [The Economist Intelligence Unit](https://www.eiu.com/n/campaigns/worldwide-cost-of-living-2020/) as one of the most expensive cities in the world in terms of cost of living. 

HDB flats are subsidized public housing in Singapore for Singaporeans. It is managed by the state House and Development Board (HDB) under a 99-year lease. The majority of the residential housing developments in Singapore are governed and developed, and home to approximately 80% of the resident population. These HDB flats are located in housing estates, which are self-contained satellite towns with schools, supermarket, shopping malls, community hospital, community centres, clinics, hawker centres, sports and recreational facilities etc. Every housing estate includes LRT stations, MRT stations and bus stops that link residents to other part of the city. Because of the scarcity of land in Singapore, HDB flats are built as apartment-style units, often in high rise buildings so as to make efficient use of space.

In the early years, owners were not able to resell their flats but this changed in 1971. Today if an owner has been living in for at least the Minimum Occupation Period (MOP), typically five years, the HDB flat can be sold.  This is the heart of this project – HDB flat resale market. 

Purchasers of resale HDB flats are usually those not eligible to purchase the Build-To-Order (BTO) flat (singles above 35 yo) or they are looking for a HDB flat that better suits their requirements. Obviously when choosing a flat there are many considerations that impact the potential price: town areas, nearby amenities/ facilities, flat size, flat type etc. 

In order to make life easier for those buyers, I have decided to study and research the Singapore HDB resale market, explore the key factors that affecting the resale price and provide a price prediction tool as well as make some recommendations for people who are planning to buy a HDB in near future.

## Executive Summary

I am a freelance data scientist and have selected this project out of self-interest. 

My __problem statement__ for this project is __how might I be able to use a regression machine learning model to build a price prediction tool to better predict the HDB resale price for those using the tool.__ 

My main __target audience__ of this project is the customer who is __interested to purchase the resale HDB flat__, as well as the __current owner__ that hopes to sell their HDB flat and the __property agents/agencies__.

The original dataset for this project is obtained from the data.gov.sg website. The dataset contains information on all the resale HDB flat transactions for __January 2017 - October 2020__, more than __80,000 resale HDB data and 11 features__.

__My goal__ for this project is to create both a high performing and generalizable regression model for predicting HDB resale price, using various regression machine learning techniques, feature engineering, feature selection and regularization. Then, I will be looking to evaluate the performance of the models using:

1. __Root mean squared error (RMSE)__ - the measure of the difference between values predicted by the model and their actual values. Lower values of RMSE indicate better fit.

2. __R-Squared (accuracy)__ - a statistical measure between 0 and 1 which calculates how similar a regression line is to the data it's fitted to. If it's a 1, the model 100% predicts the data variance; if it's a 0, the model predicts none of the variance.


## Data source:
- [Singapore hdb resale flat data](https://data.gov.sg/dataset/resale-flat-prices)
- [Singapore mrt coordinates](https://www.kaggle.com/yxlee245/singapore-train-station-coordinates)
- [Singapore shopping mall list](https://en.wikipedia.org/wiki/List_of_shopping_malls_in_Singapore)
- [Singapore List of Government Markets Hawker Centres](https://data.gov.sg/dataset/list-of-government-markets-hawker-centres)
- [Singapore Listing of All Active Pre-School Centres](https://data.gov.sg/dataset/listing-of-centres)
- [Singapore Top Primary School](https://www.salary.sg/2020/best-primary-schools-2020-by-popularity/)

## Table of Content:
1. EDA - Original Dataset - 2017 onward <br>
    1.1 Data Acquisition and Exploration <br>
    1.2 Univariate Plots <br>
    1.3 Checking for Homoscedasticity <br>
    1.4 Statsmodels <br>
2. Data Wrangling and Feature Engineering - 2017 onward <br>
    2.1 Data Cleaning <br>
    2.2 Feature Engineering <br>
    2.3 Checking for Homoscedasticity <br>
    2.4 HDB Geospatial <br>
3. EDA, Preprocessing and Feature Selection - 2017 onward <br>
    3.1 Scatter Plots of Numerical Features <br>
    3.2 Histograms of Numerical Features <br>
    3.3 Histograms and Boxplots of Categorical Features <br>
    3.4 Encoding <br>
    3.5 Correlation Heatmap of Resale Price with All Features <br>
    3.6 Pairwise Correlation <br>
    3.7 Low Variance <br>
    3.8 Recursive Feature Elimination Cross Validation <br>
    3.9 CART and Boosted Feature Importance <br>
    3.10 Permutation Feature Importance <br>
    3.11 Final Feature Selection <br>
4. Model Selection and Tuning - 2017 onward <br>
    4.1 Model Preparation <br>
    4.2 Conclusion <br>
    4.3 Benefits of the model <br>
    4.4 Limitations of the model <br>
    4.5 Future Work <br>
    4.6 Model Improvement <br>
    4.7 Web Application Improvement <br>
    
__Data Dictionary:__

1. __month (object)__: This feature shows the date of the HDB sold in the format of year and month.

2. __town (object)__: This feature shows the geographic location of the town that the HDB located in Singapore. There are 26 HDB towns in this dataset: ANG MO KIO, BEDOK, BISHAN, BUKIT BATOK, BUKIT MERAH, BUKIT PANJANG, BUKIT TIMAH, CENTRAL AREA, CHOA CHU KANG, CLEMENTI, GEYLANG, HOUGANG, JURONG EAST, JURONG WEST, KALLANG/WHAMPOA, MARINE PARADE, PASIR RIS, PUNGGOL, QUEENSTOWN, SEMBAWANG, SENGKANG, SERANGOON, TAMPINES, TOA PAYOH, WOODLANDS and YISHUN.

3. __flat_type (object)__: There are 7 different HDB flat types in Singapore: 1 Room, 2 Room, 3 Room, 4 Room, 5 Room, Multi-Generation and Executive. Flat type determines the number of rooms and amenities inside the house.

4. __block (object)__: This feature shows the number of the HDB flat in each town.

5. __street_name (object)__: This feature shows the HDB flat's street name in each town.

6. __storey_range (object)__: This feature shows the range of the building level of the HDB flat. The storey is range from floor 1 to floor 51. Each of the storey_range consists of 3 floors such as 01 TO 03, 04 TO 06 etc.

7. __floor_area_sqm (float)__: This feature shows the floor size of the HDB flat in the unit of square meter.

8. __flat_model (object)__: This feature shows the type of models of the HDB flat. There are 20 HDB flat models in this dataset: Improved, New Generation, DBSS, Standard, Apartment, Simplified, Model A, Premium Apartment, Adjoined flat, Model A-Maisonette, Maisonette, Type S1, Type S2, Model A2, Terrace, Improved-Maisonette, Premium Maisonette, Multi Generation, Premium Apartment Loft and 2-room.

9. __lease_commence_date (integer)__: This feature shows the start date of the lease of the HDB flat in the format of year.

10. __remaining_lease (object)__: This feature shows the remaining lease of the HDB in the format of year and month. The total lease of Singapore HDB flat starts with 99 years.

11. __resale_price (float)__: This feature shows the price of the HDB flat sold in Singapore dollar. This will be the target variable.

## Conclusion

As you can see from the work above, I have undertaken comprehensive analysis.

In summary, I assessed the EDA, undertook feature engineering/selection and ran regression machine learning models to select the most predictive model. 

As a result, I have successfully created a model that is able to predict the HDB flat `resale_price` in Singapore. __The Gradient Boosting Regressor__ has achieved satisfactory results __(RMSE: \$17,815, R-squared: 98.66\%)__ for HDB resale price prediction compare to the base model (Linear Regression) results __(RMSE: \$35,954, R-squared: 94.54\% - under section 2.3: Checking for Homoscedasticity)__. 

__The Gradient Boosting Regressor model is superior to a simple Linear Regression__ because Gradient Boosting is a type of machine learning boosting that relies on the intuition that the best possible next model, when combined with previous models, minimizes the overall prediction error. The key idea is to set the target outcomes for this next model in order to minimize the error. But Linear Regression establishes the relationship between two variables using a straight line and it doesn't consider the error.

The __top 5 features__ that impact the resale price based on the Gradient Boosting Regressor feature importance scores are:
1. __floor_area_sqm__ – it is obvious that the bigger the space, the greater the price of the HDB flat,
2. __dist_to_city__ – the nearer to the city, you will able to access to the better amenities and facilities,
3. __remaining_year__ – the longer the lease, you will able to apply for larger loan from the HDB or banks and the condition of the HDB flat would be relatively better,
4. __mean_floor__ – the higher floor, you will have better view, cooler, less noise, less mosquitoes etc,
5. __dist_to_pri_sch__ – Singapore citizen who live within 1km of the school will be given priority to get into the primary school. Most of the Singaporean families will tend stay near to the top primary school so that their children will able to get into the school.
    
I have also created a user-friendly __web application__ that require the user to select the features of their preferred HDB flat. Then the back-end application will generate the predictive price to provide the real time prediction.

## Benefits of the model

1. The predictive price is presented as a range e.g., the predicted price is \$450,000, the range is between \$430,000 to \$470,000,
2. User can use this model to buy or sell the HDB (DIY) without approaching the agent, can save their money to pay the agent fee.
2. Multiple features to improve price prediction,
3. User-friendly web application,
4. Can be used equally by buyer, seller, property agents or agencies and others.
    
## Limitations of the model

1. The model is only able to predict the price for the nominated features, but the user is not able to enter the target price and then receive available HDB resale flats for that price and the associated features,
2. This model is not applicable for other property price prediction because of the different features,
3. Additional datasets not included may have been useful eg. GDP, population density of the HDB estates.
    
## Future Work

Although my model has generated a reasonable low RMSE and high R-squared value (compare to baseline model), it should be noted that the real HDB flat resale price is much more complex and the list of the features, understandably is not an exhaustive list.

There are many other factors should be taken into consideration such as overall flat conditions, overall Singapore economic conditions, government housing regulation or prospective future development at different town. In addition, it would be better if we have the information of the HDB flat original value when it was first sold.

People's individual motivations will impact the price they are prepared to pay e.g., if I work in the West, I am likely to be prepared to pay more for HDB flat in the West than someone working in the East as we both factor in convenience in regards to travel in our price expectation. For someone working and living in the west, the travel time from home to CBD may not be of high priority, since they will spend most of their time within their neighbourhood.

Likewise, other social factor influences the price e.g., a neighbourhood may be undergoing upgrading new shopping mall, facilities etc and making it a "cooler" place to live. None of the datasets included capture such an issue.

## Model Improvement

Some specific elements to add to improve the predictivity of the model include:
1. Additional features such as flat condition rating, GDP and population density of the HDB estates,
2. Scrape more data from the different resources such as PropertyGuru or 99.co website,
3. Retrain the model weight as and when new data added,
4. Train a new model if new feature/s added,
5. Tuning the hyperparameters of the machine learning model to discover the parameters of the model that result in the most skillful predictions.

## Web Application Improvement

Potential improvement for the web application includes:
1. Reduce the number of user inputs by having automatic default selection e.g., when 4-Room flat is selected, the number of bedrooms and bathrooms will be filled automatically,
2. Map driven selection,
3. Visual display and details of the facilities and amenities around the area.

Overall, the model is fit for purpose. It predicts price with a reasonable RMSE, is easy to use and processes the result quickly. More data and features would improve the model but I am pleased to recommend this model to the future HDB purchasers - like myself.# Singapore-HDB-Resale-Price-Prediction
