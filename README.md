# goodbooks-10k-recommender

### Description
XGBoost Book Recommender system using goodbooks-10k dataset (binary ratings ver.)  
This recommender system provides the top 10 recommended books by the user. If recommender system doesn't have any information about the user, recommend 10 of the most popular books (books with high ratings).


### Usage
1. Download dataset  
You can download dataset here : https://www.kaggle.com/zygmunt/goodbooks-10k

2. Preprocessing  
You should run .ipynb files under preprocessing folder before train.

3. Run Train / Predict / Recommend  
mode : train, predict, recommend  
user_id : user_id (only for recommend mode)  
~~~
./run.sh {mode} {user_id}
~~~
- train : train xgboost model
- predict : evaluate xgboost model performance(rmse, accuracy)
- recommend : recommend top 10 books for user(user_id)
