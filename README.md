# ML project: Feature Engineering and Classification

## Basic info

* 2 class classification based on 35 feature values
* 500k training set

  * 50K useless data
  * unbalanced: 8:1 (class0: class1)
* 200k test set

  * almost balanced

## Dataset preprocess

* Use PCA to recoginize the useless dataset (50k)
* Some early submissions tell us we have almost balanced test set -> use balanced val set
* Use 10000 samples for validation. (5000 class0 / 5000 class1)
* Use random forest to get the importance of each features, used in NN training
* Use balanced train set (repeat class1 for 8times)
* Standardize all the features

**The original seed for splitting validation set is missing, But it should not affect the final test score for more then 0.0005**

## Training

* Use Neural Network

  * Arch: custom arch named Feature Expander
  * Basically an arch for expanding a feature vector into higher/more NN friendly feature space.
  * Use Linear Projection/Residual/MLP with higher hidden dim. Which are widely used in recent SoTAs.
  * Use dropout
* Use Prodigy Optimizers:

  * [[2306.06101] Prodigy: An Expeditiously Adaptive Parameter-Free Learner (arxiv.org)](https://arxiv.org/abs/2306.06101)
* Use cosine annealing lr scheduler

  * [[1608.03983] SGDR: Stochastic Gradient Descent with Warm Restarts (arxiv.org)](https://arxiv.org/abs/1608.03983)
* Use Cross Entropy Loss with weight. Since class1 have more repeats which caused some kind of "overfit", we actually need higher loss weight for class0.
* Use label smoothing
* Best settings

  * hidden dim 128
  * keeped features 24 (24 features high higher importance in random forest)
  * CE loss weight: [1.1, 1.0]
  * label smoothing: 0.1
  * 100 epoch/batch size 2048/lr=1.0, lr_min=0.01
  * dropout rate = 0.5
  * best seed: 3407~3422 (ignore 3410 for explosion)

## Inference

* Combine lot of trained networks together (same arch, different seed)

  * Get the output of softmax of each network, add them together.
* Ignore the network models which have some "explosion" effect during training

  * Usually Introduced by dropout + Prodigy optimizer

## Extra infos:

Random Forest with top 9 importnace features can get 0.856

NN can always get 0.8735↑ need some lucks to get over 0.8739
