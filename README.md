HOUSE PRICE PREDICTION:
-----------------------

This project is on House price prediction where we are trying to predict prices of house.

I used some basic analysis like - how many different bedrooms in houses, house prices according to locations (latitude and longitude).

Tried linear models (linear regression and ElasticNet) but it did not give good result.

Reason I believe might be - collinearity between independent variables and also some outliers.

For tree models, results are very good.

Even after tuning parameters results were same so, went with default parameters.

Random Forest, XGB, Gradient Boost performed well but training time is good for XGBoost.

Also used Voting Regressor with XGBoost and Random Forest Regressor, it also gave similar results but took more training time.

About Scaling - with scaling the results were not very good so, did not use scaling.
Also tried removing outliers (which I found by plotting bar plot) but the results were same

goal is to see how model is performing on different unseen samples so used cross-validation to see performance of model on different set of samples also estimate skills of model on unseen data it is also used to see if the model is suffering from overfitting.


Included MLflow, using which checked many parameters for models
How to do ? 
On terminal --> python house_price_mlflow.py 800 5 0.1 (these are parameters for trees) likewise, it can be done for various models and also carry out analysis between different parameters.
Experiments done for linear model with alpha and l1_ratio as parameters and with alpha = 0.001 and l1_ratio = 0.8 and among the 6 experiment I got highest r2_score of 68%.  

![image](https://github.com/LearningPratik/house_price_prediction/assets/139999671/e3c358d2-47e1-4258-825f-17896e8f6494)

Experiments done for boosting algorithm with estimators, depth and learning rate, best parameters among various experiments were 800 estimators, 5 depth of tree and 0.1 learning rate

![image](https://github.com/LearningPratik/house_price_prediction/assets/139999671/5cc4df8e-f1c9-4779-a9f7-5ce33451618e)

Also used FastAPI to serve it as API.

![image](https://github.com/LearningPratik/house_price_prediction/assets/139999671/79cce46a-3dcb-4933-a7ec-98da0e0ca59a)


<b>Feature importance :</b>

![image](https://github.com/LearningPratik/house_price_prediction/assets/139999671/60844d68-a77f-46c8-9606-8a6a7299602f)
