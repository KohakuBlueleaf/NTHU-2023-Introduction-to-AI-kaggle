# info for best models

### 0.87378

* 15 seed ensemble (15seed with hidden dim 128 can stably achieve 0.873)
* hidden dim 128
* structure: Feature Expander (features=24)
* seed = 3407~3421
* standardized, balanced
* CE loss weight [1.1, 1.0]
* label smoothing = 0.1
* 60epoch/bs 2048/ cosine annealing/eta_min=1e-2/lr=1
* Prodigy optimizer
* Dropout = 0.5

### 0.87251

* Use balanced val
* val f1: 0.8764
* val acc: 0.8764
* Single model
* hidden dim 192
* structure: Feature Expander (features=24)
* seed = 0 (seed_everything)
* standardized, balanced
* add CE loss weight [1.05, 1.0]
* add label smoothing=0.2
* 50 epoch/batch size 2048/cosine anneal/eta_min=1e-2/lr=1
* prodigy optimizer
* Only keep first 24 features. (Ordered by random-forest importance)
* Dropout = 0.5
* Balanced Val, unbalanced, test

### 0.87241

* val f1: 0.87894
* val acc: 0.8767
* Single model
* hidden dim 128
* structure: Feature Expander (features=24)
* seed = 0 (seed_everything)
* standardized, balanced
* add CE loss weight [1.02, 1.0]
* add label smoothing=0.1
* 50 epoch/batch size 2048/cosine anneal/eta_min=1e-2/lr=1
* prodigy optimizer
* Only keep first 24 features. (Ordered by random-forest importance)
* Dropout = 0.5

### 0.87198

* val f1: 0.8807
* val acc: 0.8773
* Single model
* hidden dim 128
* structure: Feature Expander (features=24, weighted_residual=True)
* seed = 0 (seed_everything)
* standardized, balanced
* add CE loss weight [1.0, 1.05]
* add label smoothing=0.1
* 50 epoch/batch size 2048/cosine anneal/eta_min=1e-2/lr=1
* prodigy optimizer
* Only keep first 24 features. (Ordered by random-forest importance)
* Dropout = 0.5

### 0.86991

* val f1: 0.87637
* val acc: 0.87340
* Single model
* hidden dim 128
* structure: Feature Expander(features=24, weighted_residual=True)
* seed = 0 (seed_everything)
* standardized, balanced
* add CE loss weight [1.0, 1.2]
* add label smoothing=0.1
* 10 epoch/batch size 64/cosine anneal/eta_min=1e-2/lr=1
* prodigy optimizer
* Only keep first 24 features. (Ordered by random-forest importance)

### 0.86805

* val f1: 0.87532
* val acc: 0.87180
* Single model
* hidden dim 128
* structure: Structure1(features=34, crossing=False)
* seed = 0 (seed_everything)
* standardized, balanced
* removed col 20, which caused 5 cluster in PCA.
* add CE loss weight [1.0, 1.2]
* add label smoothing=0.1
* 10 epoch/batch size 64/cosine anneal/eta_min=1e-2/lr=1
* prodigy optimizer

### 0.86685

* val f1: 0.874
* val acc: 0.8702
* Single model
* hidden dim 128
* structure: Structure1(features=34, crossing=False)
* seed = 0 (seed_everything)
* standardized, balanced
* removed col 20, which caused 5 cluster in PCA.
* add CE loss weight [1.0, 1.2]
* add label smoothing=0.1
* 10 epoch/batch size 64/cosine anneal/eta_min=1e-5/1r=1e-3
* adamw optimizer

### 0.86593

* val f1: 0.873
* val acc: 0.8717
* Single model
* hidden dim 128
* structure: Structure1(features=35, crossing=False):
* seed = 0 (seed_everything)
* standardized, balanced
* 10 epoch/batch size 64/cosine anneal/eta_min=1e-5
* adamw optimizer
