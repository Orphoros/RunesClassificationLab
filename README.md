# Runes Image Classification with Machine Learning

> In association with [Astraeus-](https://github.com/Astraeus-)

In this document we go over decision choices we made during the project and the reasoning behind them. We also go over the different phases of the project and the results we got from them. We introduce our project structure and we discuss which model we deployed.

## Iterations

| Iteration | Description |
| --- | --- |
| `0` | **Manual** image classification. We extract 2 features, and use both of them. We measure the distance between our sample and training data and choose the closest one. This is done with a custom algorithm. |
| `1` | **kNN** image classification with manual feature selection. We extract 23 base features out of which we select 6 to use based on the Pearson correlation matrix. |
| `2` | **kNN** image classification with automatic feature selection using the chi2 algorithm. We use the same 23 base features from iter `1`, out of which we select 20 to use. |
| `3` | **Naive Bayes** image classification with automatic feature selection using the chi2 algorithm. We use the same 23 base features from iter `1`, out of which we select 20 to use. |
| `4` | **SVM** image classification with automatic feature selection using the chi2 algorithm. We use the same 23 base features from iter `1`, out of which we select 20 to use. |
| `5` | **Decision Tree** image classification with automatic feature selection using the chi2 algorithm. We use the same 23 base features from iter `1`, out of which we select 20 to use. |
| `6` | **Random Forest** image classification with automatic feature selection using the chi2 algorithm. We use the same 23 base features from iter `1`, out of which we select 20 to use. |
| `7` | **Neural Network MLP** image classification with automatic feature selection using the chi2 algorithm. We use the same 23 base features from iter `1`, out of which we select 20 to use. This is the iteration for a custom picked model. |
| `8` and `9` | **kNN** as in iter 2 with a different scaler (Standard Scaler from iter 3.3, and Max Abs Scaler from iter 3.4 ) |
| `10` | **SGD** image classification with automatic feature selection using the chi2 algorithm. We use the same 23 base features from iter `1`, out of which we select 20 to use. |

### Feature Extraction (Step 1)

#### Iter 0 Initial feature extraction

In our initial feature extraction phase we focused on extracting two features. The number of black pixels in the image as well as the number of regions in the images. We wanted only the most dominant features, and not a lot of them, so we can easily classify the images manually.

We accomplished this by converting all the images to grayscale, removing the potential alpha channel and applying binary erosion to remove noise and make the images clearer.

#### Iter 1 Extracting more features

Based on the lack of success of our initial model, we extracted a lot more features from the images. We cleaned our input data by cropping the images based on the drawn input and scaling the selection to the same size, thus making sure that the drawn image takes up the whole image (in case the image is drawn in the top left corner). We then added a border so that the drawn image does not touch the borders. This is important so that we can get the accurate number of holes. We also inverted the colors to get better regionprop results. We extracted all regionprop features where the returned value is a single number value.

We also added custom smart features. We checked for vertical and horizontal symmetry of the image by calculating the symmetry error by overlapping halves of the image. We also measured the amount of information in 9 different regions of the image by creating a 3x3 grid and calculating the number of black pixels in each region.

In this step, we got 23 features in total.

### Feature Analysis (Step 2)

#### Iter 0

We worked with only 2 features. There are no missing features (NaN or null values). From the Pearson correlation matrix we can see that the two features are not correlated at all.

From the boxplot on the sum of black pixels we can see that there are some large outliers. These images are just scribbles. We can also see a lot of smaller outliers slightly above the median. This is because the stroke thickness of the image can be changed. These are most likely those images with a thicker stroke.

Based on the histogram of the number of regions, we can see that there are some outliers. There are images with a huge number of regions. These are most likely scribbles. All in all, the number of regions are very consistent and clear and the number of outliers is very low.

We have around 120 images per label. This is a good amount of data to work with.

When we look at our pairplot when we plot the two features against each other, we can see that there are some groups forming. However, the groups are not very clear. We can also find our major outliers here.

We select and continue using both of these features.

#### Iter 1

Having 23 features, we can see more correlations. When looking at the Pearson correlation matrix we can observe strong, positive correlations between some features, low, negative correlations between some features and no correlation as well.

In our pairplot we can see much clearer groups of labels in some of the features. This is great as it means we can use these features to classify our images.

When looking at the hole count, we can observe that the hole count is very consistent. There are little to no outliers. When there are outliers, their count is really small. This proves that cropping, stretching the image and then adding a border makes our image better classifiable.

The violin plots show us that most of our features are very consistent. There are some outliers, but those are because of adjustable stroke width. Where there are large outliers (further away from the median), those are most likely scribbles, and their number is few.

What is interesting is the vertical and horizontal symmetry. Looking at the results and the original images, and considering their symmetry, it seems that the symmetry error threshold is at around 30%. Those images, whose symmetry error is lower then 30% are likely to be symmetric. Above 30% they are not. This is a good feature to use. The bigger the median, the less certain we are about the symmetry of the image. For some runes, like `Sun`, it is very clear that they are not symmetric at all, as both horizontal and vertical symmetry is at 40%, and the distribution is very small. For some runes, like `Tyr`, the distribution is much larger, making it less certain about symmetry.

We manually selected 6 features with a negative or low correlation to make sure we have different, non-supportive features.

#### Iter 2

In this iteration we have the same data as in iter 1, but now we want automatic feature extraction. With the chi2 algorithm, we selected 20 features to use. We decided, that we should have a large amount of features and dropping 3 would be enough to make sure we have a good amount of features, while also getting rid of the least important ones.

### Preprocessing (Step 3)

#### Iter 0

Since we are preparing data for our custom, manual classification, we want very clean data. We make sure that there are no outliers at all for the regions. We want the region count to be the same across all images per category. This wouldn't be a good idea for a machine learning models because of overfilling, but for us it makes our work easier. We only aimed to remove the big outliers (the scribble images). We did this with our region count values.

#### Iter 1

We aim to keep some outliers so that the models we train on this dataset won't be overfitting. We aim to remove only the scribble images. We do this by removing images with a too high area (scribbles have a high area). We also remove images with 0 eccentricity. These images have incorrect data.

#### Iter 2

Just like in iter 1, we want to keep some outliers so that our model is not overfitted. This time, however, we removed more outliers. We removed images that have a hole count larger than 3. The max hole count is 2 (based on our image dataset) and we decided to add one more on top of it in case of errors. We also remove scribbles whose area is too big.

### Training and Testing (Step 4)

#### Iter 0

We created a custom algorithm to classify our images. We did this based on two features: back pixel sum and region count. Our algorithm works by finding the closest match in its training dictionary lookup table. 

When we provide the training data for our model, we fit it by creating a lookup dictionary. We do this by creating an entry (key) for each unique label we have. Then, we average out all the features (value) for the label. We do this to have a generic description for all labels.

For prediction, our model first tries to match the region count. We then filter the dictionary entries based on the matches. If no match is found, we just use the whole lookup table. Then, after the filtering, using minimum distance, we look up which entry is the closest from the lookup table to our sample.

From the confusion matrix we can observe that our model is 38% accurate. This is likely this low because we only used 2 features and we didn't scale our data. It could predict some runes very well, like `Spear` or `Bow`, but overall, it had a lot of confusion.

#### Iter 1

We used a kNN model in this run for our manual feature selection (for 6 features). We trained our model on 75% of our data, using the remaining 25% to test our model. Based on our findings from iter 0, we used a min-max scaler for kNN, as this is also a distance-based model, like our manual one.

We used Grid Search to find the best hyperparameter for our kNN model. We also used k-fold with 5 folds to check and prevent overfitting when finding the best parameters.

Looking at the confusion matrix, we can see that we got a 73% overall accuracy. There is now less confusion. However, some labels still often get confused. This is likely because we didn't select the best features and we also need more features to use.

Our cross validation scores confirm that there isn't a lot of overfitting. We calculated the maximum difference and standard deviation between our 5 folds. The smaller the number the better.

#### Iter 2

After iter 1, we concluded that we need more features. In this iteration we used 20 features automatically selected by the chi2 algorithm. Just like in iter 1, we created a train-test split, scaled the data and used grid search with 5 folds.

This time, our confusion matrix confirms our theories for improvement from iter 1. We got a 93% accurate model with little to no confusion. Only 2 runes, Ash and Wealth got a bit confused with each other. The max difference and standard deviation for the folds are also lower, confirming that this model is less overfitted than the one in iter 1.

#### Iter 3

For this iteration and onwards we will keep using the data from the automatic feature selection, as iter 2 confirmed that our dataset is great for machine learning models. Now, we try the Naive Bayes model on our iter 2 dataset. Again, we used train-test split and grid search with 5 folds for cross validation. We did not use a scaler, as it is generally not recommended for this type of model. Thus, for this iteration, we decided not to use one.

The confusion matrix shows a lower accuracy, 88%, which means, that Naive Bayes isn't the best for our dataset. The same two runes, Ash and Wealth, got confused more and some others got slightly confused as well. However, looking at the maximum diff and standard deviation for our folds, we get the same scores as in iter 2. Thus, our model's performance is similar.

#### Iter 4

Here we applied an SVM model to our dataset from iter 2. We used a train-test split, grid search with 5 folds to optimize our model. We used a min-max scaler as it is recommended for this model.

Our confusion matrix shows a very high, 97% accuracy! This time only Wealth and Ash got confused and nothing else! When we look at the overfitting scores, we get the lowest scores. This means, that an SVM model is better than a kNN for our dataset.

This model turned out to be the best model, so this is what we deployed to the web server.

#### Iter 5

In this iteration we tested Decision Tree for our iter 2 dataset. We did the same steps as in the iterations before. We did not scale the data as it was not recommended for this model.

We got a lower score compared to our best model, SVM, with an 87% accuracy. The overfitting metrics share the same results as in iter 2 for kNN.

#### Iter 6

We used Random Forest for this iteration based on the dataset from iter 2. We did the same steps as in the iterations before, but we also min-max scaling.

Our accuracy score is 94%. This model is slightly less overfitting than our kNN model from iter 2, but is more overfitting than the SVM model from iter 4.

#### Iter 7

Lastly, for this iteration we chose a custom model from scikitlearn. We chose a neural network model, MPL Classifier. We wanted to try our an ANN model as well for our iter 2 dataset.

We used min-max scaling (as recommended) and used grid search with 5 folds for cross-validation and finding the best hyperparameter. We got 96% accuracy with little to no overfitting. This model is the 2nd best, after SVM. Only the runes Wealth and Ash got confused again.

## Conclusion

Our SVM model performed the best. This is because we scaled our data, and we used a lot of features where we selected the best ones. Further improvements would be creating more custom features that distinguish Wealth and Ash runes, as they always got confused in all models.