
# Recommendation System for New York Times

## Project Description

In this project, I built and optimized an article recommendation system for New York Times to give top recommendation articles to visitors, based on 350k+ page visit records from visitors from 2018-07-01 to 2018-10-01. The final average user precision and recall are of .

First, I split the whole data set into training(65 days)/validation(13 days)/test sets(13 days), and performed basic data cleaning, processing and exploratory data analysis on the training set. I split the training set further into five parts with same time span (13 days). For each interval, I analyzed the distribution of page visits for individual articles and for individual visitors, and components of page visits with respect to previous/new articles and former/new visitors.  <br/>
Second, I performed natural language processing (NLP) on the article titles and categories and built the TF-IDF matrix for both training set and validation set, computed the cosine similarity between new articles and previous articles.
Third, I recommended similar and new articles to former visitors and popular and new articles to new visitors. 
Finally, I optimized the recommendation by optimizing the number of articles recommended and the features in TF-IDF feature matrix. 


* All the project files are contained in this repo
* Tools/Languages: `Python(Pandas, NumPy, Tensorflow, Scikit-Learn)`



## File Description

#### 1. `Resources`

* `cumulative.csv` contains the raw data obtained from Kaggle
* `cleaned_data.csv` contains the cleaned data


#### 2. `Static`
* `ETL_EDA.ipynb` contains the Python code for data cleaning, data processing (scaling/dimensionality reduction/resampling) and exploratory data analysis
* `pca_lda_test.ipynb` contains the Python code to check the performances of models based on PCA/LDA processed components
* `logistic_regression.ipynb`/`SVM.ipynb`/`random_forest.ipynb` contain the Python code for model training and model optimization process
* `logistic_regression.sav`/`SVM.sav`/`random_forest.sav` contain the optimized models
* `Test.ipynb` contains the Python code to check the performance of final optimized model on test set





