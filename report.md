# Machine Learning Engineer Nanodegree
## Capstone Project — Online News Popularity
Chuanyue You  
January 17th, 2018

## I. Definition

### Project Overview 

Due to web expansion, the Internet, gradually takes the place of traditional media like newspapers, 
becomes the major medium for people to read and share news. The study of online news popularity, 
therefore, is an important topic to understand people’s preference of news. 
The popularity of a news is indicated by the number of shares in social networks.<br><br>
This project intends to find the best model and set of features, 
trained using Online News Popularity dataset from UCI Machine Learning Repository, 
to predict the popularity of online news.

### Problem Statement

The goal is to train a model that can predict the popularity of online news. 
One common approach is to train a regression model to precisely predict the number of shares in social networks. 
This approach, however, is difficult for this project due to the high variance of target variable. 
Instead I discretized target variable into two classes (popular and unpopular) using median value as decision threshold, 
which resulted in a balanced class distribution, and trained classification models to predict the popularity. <br><br>
The final model is expected to predict the popularity of unseen online news with reasonable accuracy.

### Metrics 

- **Accuracy**: (tp+tn)/(tp+fp+tn+fn)<br>
Accuracy is a good performance measure to see the overall performance of classifier, 
especially for balanced dataset like online news data.
- **F1-score**: 2*(precision*recall)/(precision+recall)<br>
The F1-score conveys the balance between the precision and the recall.

## II. Analysis 

### Data Exploration

The online news popularity dataset contains 39,644 articles (samples) from the Mashable website  (www.mashable.com) 
with 61 attributes (58 predictive attributes, 2 non-predictive, 1 goal field) respect to 6 aspects of online news, 
namely words, links, digital media, publication time, keywords and natural language processing. 
Among 58 predictive attributes, 44 of them are numerical data and the rest are categorical data.<br>
![dataset](https://github.com/ChuanyueYau/OnlineNews/blob/master/report_images/dataset.png)
<br>As it can be expected based on the above descriptions, some of attributes are the ratios range from 0 to 1, which means some outliers that are out of this range should be removed during the preprocessing.

### Exploratory Visualization

In order to clearly visualize the distribution of all features and get rid of the impact of extreme values, 
I only visualized the part that are less than 98 percentile.<br>
![distribution](https://github.com/ChuanyueYau/OnlineNews/blob/master/report_images/before_transformed.jpg)
<br>From the plot above, we found these features are highly skewed:<br>
'n_tokens_content','num_hrefs','num_self_hrefs','num_imgs','num_videos','kw_max_min','kw_avg_min','kw_min_max',
'kw_max_avg','self_reference_min_shares','self_reference_max_shares','self_reference_avg_sharess' 

### Algorithms and Techniques 

- **Machine Learning Algorithms**<br><br>
**Random Forest**:
Using random forests to provide land classification is a good example of the applicability of classifier ensembles to remote sensing.<br>
Strengths: Random Forests are fast, flexible, and represent a robust approach to mining high-dimensional data. 
They perform well even in the presence of a large number of features and a small number of observations.<br>
Weakness: It takes time to make predictions.<br>
Since our dataset has 58 features, this model might handle our data well.<br><br>
**Gradient Boosting**:
Gradient boosting can be used in the field of learning to rank. 
The commercial web search engines Yahoo and Yandex use variants of gradient boosting in their machine-learned ranking engines.<br>
Strengths: 
1.Natural handling of data of mixed type(heterogeneous features)
2.Predictive power 
3.Robustness to outliers in output space(via robust loss functions)<br>
Weakness: Scalability, due to the sequential nature of boosting it can hardly be parallelized and therefore are time-consuming. 
The model alos has more parameters need to be tuned. <br>
Since our dataset has both numerical and categorical data type and gradient boosting model usually yield accurate results, 
this model might be a good candidate for the problem.<br><br>
**AdaBoosting**:
AdaBoost is a type of "Ensemble Learning" where multiple weak learners are employed to build a stronger learning algorithm. 
AdaBoost works by choosing a base algorithm (e.g. decision trees) and iteratively improving it by accounting for 
the incorrectly classified examples in the training set. We assign equal weights to all the training examples and 
choose a base algorithm. At each step of iteration, we apply the base algorithm to the training set and increase 
the weights of the incorrectly classified examples. We iterate n times, each time applying base learner on the 
training set with updated weights. The final model is the weighted sum of the n learners. <br>
Why the Adaboosting should be tried though we’ve tried GradientBoosting? The AdaBoost algorithm is quite 
resistant to overfitting (slow overfitting behaviour) when increasing the number of iterations. 
This parameter can be deduced through cross-validation or some other way.<br><br>
**SVM**:
SVM’s basic idea is to transform the attributes to a higher dimensional feature space and find the optimal 
hyperplane in that space that maximizes the margin between the classes. It has a regularisation parameter, 
which makes the user think about avoiding over-fitting. It uses the kernel trick, so we can build in expert
knowledge about the problem via engineering the kernel. <br>
The disadvantages is that kernel models can be quite sensitive to over-fitting the model selection criterion. <br>
Nevertheless, since we will test the robustness of the final model and SVM always provide advanced results, 
we should also try SVM.

- **Feature Selection**<br><br>
**Recursive Feature Elimination**: Given an external estimator that assigns weights to features 
(e.g., the coefficients of a linear model), the goal of recursive feature elimination (RFE) 
is to select features by recursively considering smaller and smaller sets of features. First, 
the estimator is trained on the initial set of features and the importance of each feature 
is obtained either through a coef\_ attribute or through a feature\_importances\_ attribute. 
Then, the least important features are pruned from current set of features. That procedure is 
recursively repeated on the pruned set until the desired number of features to select is eventually reached.<br>
RFE can remove some noisy or redundant features for learning algorithms, therefore increasing the 
its performance and reducing the training and predicting time.

### Benchmark

Instead of choosing naive predictor, which simply predicted all samples to be popular news and got 53.36% accuracy
and 0.6959 f1-score, as benchmark model, I found a related publication from Stanford Machine Learning Report 
and used its established result as benchmark. [He Ren and Quan Yang, ‘Predicting and Evaluating the Popularity of Online News’, 
Standford University Machine Learning Report, 2015] <br><br>
It measured the performance by using accuracy score and recall. Depending on different problems to address, 
the performance measures can vary a lot. Recall is a good measure if the purpose is to extract the popular 
news as many as we can. However, I preferred f1-score more since it conveys the balance between the precision 
and the recall, focusing on both the quantity and the quality of extracted popular news. The best accuracy 
score attained by the benchmark is 69%.<br>
<img src="https://github.com/ChuanyueYau/OnlineNews/blob/master/report_images/benchmark.png" width="600">



## III. Methodology

### Data Preprocessing

- **Feature transformation**<br><br>
Conduct log transformation on skewed continuous features:<br>
'n_tokens_content','num_hrefs','num_self_hrefs','num_imgs','num_videos','kw_max_min','kw_avg_min','kw_min_max',
'kw_max_avg','self_reference_min_shares','self_reference_max_shares','self_reference_avg_sharess'<br><br>
Feature distributions after log transformation:<br>
![transformation](https://github.com/ChuanyueYau/OnlineNews/blob/master/report_images/after_transformed.jpg)<br>
From the plot above, we can see that after log transformation, the distribution of these skewed features became normally distributed.

- **Remove outlier**<br><br>
I removed one outlier, No. 31037 instance. The reason is that this instance has three attributes,
namely ’n_unique_tokens’, ’n_non_stop_unique_tokens’ and ’n_non_stop_words’, that are far out of range. 
Since all these three attributes represent ratio, which should range from 0 to 1. This instance, however, 
has value of 701, 650 and 1042 for these three attributes respectively.

- **Normalize numerical features**<br><br>
Employed MinMaxScaler to normalize all the numerical features. Normalization ensures that each feature is 
treated equally when applying supervised learners.

- **Shuffle and Split Data**<br><br>
I split dataset into training, validation and testing sets. I kept the testing set unseen during the 
entire process of model training, selection and parameter tunning, only used this part of data at the 
very end to verify the robustness of final model.


### Implementation

- **Creating a Training and Predicting Pipeline**<br><br>
Trained 4 classifiers using 5%, 20% and 100% of training data respectively. Used trained models to make 
prediction on 500 random selected training samples and entire validation sets, recorded prediction time 
and computed accuracy score and  f1-score.<br><br>
- **initial model evaluation**<br>
![model](https://github.com/ChuanyueYau/OnlineNews/blob/master/report_images/model.png)<br>
I would say the Gradient Boosting Classifier and Adaboosting Classifier are both appropriate for the task of identiying the 
popularity of online news. From the ‘Accuracy on Validation Set’, both Gradient Boosting Classifier and Adaboosting Classifier 
perform well as the training set size increased.From the 'F1-score on Validation Set' histogram we can see that 
Gradient Boosting Classifier always has highest score with different testing set size. Adaboosting also has good 
performance as the training set size increased.For the time of ‘Model Training’, Gradient Boosting Classifier requires 
second highest training time, which is OK: since our model is offline model and the dataset is static, thus we have 
enough time to train. The key is the time of ‘Model Predicting’. Apart from SVC, other models are very fast.<br><br>
The best classifiers with highest accuracy (given whole training set is trained):<br>
<img src="https://github.com/ChuanyueYau/OnlineNews/blob/master/report_images/inital_model_acc.png" width="500"><br>
The best classifiers with highest f1score (given whole training set is trained):<br>
<img src="https://github.com/ChuanyueYau/OnlineNews/blob/master/report_images/inital_model_f1.png" width="500"><br>
Nevertheless, besides Gradient Boosting Classifier, I still keep Adaboosting classifier as the best classifiers since the results for adaboosting are just slightly worse from that for Gradient Boosting Classifier

### Refinement

- **Feature Selection** —  Recursive Feature Elimination<br><br>
The reason I did feature selection in refinement section instead of data preprocessing section is that RFE selects the 
best combination set of features for a specific learning algorithm, therefore it would be a better approach to select 
some desired learners before using RFE to select features.<br><br>
I chose the search range of the number of features to be between 20 and 30 and selected the number of features for 
best classifiers (Gradient Boosting and AdaBoosting) that maximize the accuracy score on validation set. As a result, 
28 features were selected for Gradient Boosting and 30 features were selected for AdaBoosting.<br>
Selected features for two learners:<br>
![features](https://github.com/ChuanyueYau/OnlineNews/blob/master/report_images/features.png)

- **Parameter Tunning**<br><br>
For Gradient Boosting Classifier, I choose following parameters to tune:<br><br>
**learning_rate**: Learning_rate determines the impact of each tree on the final outcome. Lower values are generally
preferred as they make the model robust to the specific characteristics of tree and thus allowing it to generalize 
well. Lower values would require higher number of trees to model all the relations and will be computationally expensive. <br><br>
**n_estimators**: The number of sequential trees to be modeled .Usually Gradient Boosting Classifier is fairly 
robust at higher number of trees.<br><br>
**max_features**: As a thumb-rule, square root of the total number of features works great but we should check 
upto 30-40% of the total number of feature.<br><br>
**max_depth**: Max_depth is used to control over-fitting. Higher depth will allow model to learn relations 
very specific to a particular sample.<br><br>
**min_samples_leaf**: Defines the minimum samples (or observations) required in a terminal node or leaf.Used 
to control over-fitting.Generally lower values should be chosen for imbalanced class problems. In our problem, 
the class is slightly imbalanced so by adjusting this parameter, we could prevent overfitting hopefully.<br><br>
**random_state**: Set the random seed so that same random numbers are generated every time. This is important 
for parameter tuning. If we don’t fix the random number, then we’ll have different outcomes for subsequent runs
on the same parameters and it becomes difficult to compare models.<br><br>
For Ada Boosting Classifier, I choose following parameters to tune:
**learning_rate**, **n_estimators**, **random_state**: the reasons are the same as what written above.<br><br>
Instead of using GridSearchCV from scikit-learn to tune parameters, I chose another package called **Parfit**. 
Unlike GridSearchCV, which chooses parameters to maximize the performance metric on training set, Parfit can 
focus on maximizing the performance metric on training set.<br><br>
After parameter tunning, the best parameters for AdaBoosting are:<br><br> AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=0.1, n_estimators=500, random_state=42);<br><br>
best parameters for Gradient Boosting are:<br><br>
GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.05, loss='deviance', max_depth=6,
              max_features=1, max_leaf_nodes=None,
              min_impurity_split=1e-07, min_samples_leaf=3,
              min_samples_split=2, min_weight_fraction_leaf=0.0,
              n_estimators=300, presort='auto', random_state=42,
              subsample=1.0, verbose=0, warm_start=False) 

## IV. Results 
### Model Evaluation and Validation 

Performance metrics for unoptimized model and optimized model<br>
<img src="https://github.com/ChuanyueYau/OnlineNews/blob/master/report_images/optimized.png" width="500"><br>
After feature selection and parameters tunning, both accuracy score and f-score for validation dataset are 
slightly improved. The Gradient Boosting performs better than Ada Boosting on both performance measures, 
therefore it (Gradient Boosting) is selected to be the final model. Moreover, the scores for validation dataset 
and testing dataset, which is never seen by model, are almost the same, which indicate that the final model is robust.

### Justification 

Compared with benchmark model, my final result is slightly worse, roughly 2%, on accuracy score but almost same on 
recall score, which is 0.7125. The reasons might be following:<br>
- My data split method is different from benchmark. I split data into training, validation and testing sets while 
benchmark only has training and testing sets.
- The final algorithm used in benchmark is Random Forest while I used Gradient Boosting for final model.
- The benchmark model used different feature selection method, therefore the features selected for its model were 
different from mine.

Overall, my final result is very close to the benchmark model and is robust and significant enough to identify 
popularity of online news.

## V. Conclusion 

### Free-Form Visualization 

Five most important features for Gradient Boosting:<br>
![importance](https://github.com/ChuanyueYau/OnlineNews/blob/master/report_images/importance.png)<br>
By exploring the five most important features for Gradient Boosting, I found all of five features are related to 
the number of shares of some specific aspects of a news and three of them are keywords-related features, 
namely 'kw_avg_avg', 'kw_max_avg' and 'kw_min_avg'. It is quite reasonable that these shares-related features 
could influence the shares of a news articles. These three keyword-related features represent the average number 
of shares of best, average and worst keyword in an article. This is consistent with what we are expected, since 
when scanning news articles readers are more likely to be attracted by the articles with some popular keywords 
and ignore those with unpopular keywords.

### Reflection 

When I did the parameter tunning part, firstly I employed GridSearchCV, the method that we usually rely on to do 
parameter tunning. However, when I digged into the mechanism operated behind it, I found that this method only 
optimizes on the training data, which could be overfitting. We need to ensure that our model is not only performing
well on training dataset, but more importantly, it should be significant on unseen dataset. It is better practice 
to split the training and validation sets beforehand and enter the validation set as the scoring set to avoid 
confusion and bleeding over between your training and validation sets. Since the training set is used to train 
a given model, while the validation set is used to choose between models and the test set is used to verify our model.<br>
I was wondering if there’s other methods can do parameter tunning. Then I found Parfit, which maximize metric 
on validation dataset when conducting parameter tunning. It is a more advanced method compared to GridSearchCV. 
From this experience I learned one thing: before applying techniques, we should really ask ourselves: why bother 
using this technique? Are we sure about this? The two questions can help us save a lot of time and make the right decision.

### Improvement

As is seen from the result, the final model failed to pass 70% accuracy given the data set we have. There are two possible
approaches to improve the performance. 
- First, regarding to the models, even though gradient boosting is a very powerful model, there are some deep learning
models might performance better.<br><br>
- Second, regarding to the features, the dataset only extracted 58 features from news articles and our later work is
based on these features. However, there could be other valuable features in the articles, every single content could 
somehow influence the popularity of articles and these features were not covered by our original 58 features. 
Therefore, it could be helpful to include all the words in articles as additional features or process these words 
into TF-IDF measures and add to features.
