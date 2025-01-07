 

 ## I Dont know for classification

 The problem is that many classification algorithms assign a class to every sample, even if that sample is very different from the training data.

Some methodsto allow a classifier to say "I don't know":

## Single Stange Approaches

**Neural Networks:** A rejection class can be added to a neural network by generating training samples in the non-overlapping regions of hypercubes surrounding the data of each class. This allows the network to classify samples that are very different from the training data as a rejection class.

**Bayesian Classifiers:** Using Gaussian class densities, a Mahalanobis distance can be calculated to determine the probability of observing a distance greater than a certain value. If that probability is below a certain threshold, the sample can be rejected.

### Two-Stage Approaches: 

These use two models. The first model rejects samples not part of the training data. The second model is a classifier. Two rejection models are mentioned as examples:

**One-class SVM:** A one-class support vector machine with a radial basis function kernel can be used to capture arbitrary data distributions and reject samples.

**Isolation Forests:** This method isolates anomalies from the data with only a few decisions, using a low depth in trained isolation trees.

 It is generally preferable to train only one model to keep complexity low, but if the model cannot reject samples, a two-stage procedure may be necessary. The consequences of misclassification should always be considered for each application.


### References

[Blog] (https://www.imt.ch/en/expert-blog-detail/know-what-you-dont-know-en)