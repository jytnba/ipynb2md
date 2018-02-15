

```python
import numpy as np
import dicom
import glob
import os
import cv2
import xgboost as xgb
import pandas as pd
import mxnet as mx

from sklearn.model_selection import StratifiedKFold
from scipy.stats import gmean
from matplotlib import pyplot as plt

INPUT_FOLDER = 'stage1/'
n_patients = len(os.listdir(INPUT_FOLDER))
```


```python
def get_conv_model():
    model = mx.model.FeedForward.load('pretrained_models/resnet-50', 0, ctx=mx.gpu(), numpy_batch_size=1)
    fea_symbol = model.symbol.get_internals()["flatten0_output"]
    feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, numpy_batch_size=64,
                                             arg_params=model.arg_params, aux_params=model.aux_params,
                                             allow_extra_params=True)

    return feature_extractor
```


```python
def get_3d_data(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return np.stack([s.pixel_array for s in slices])
```


```python
def get_scan(path):
    sample_image = get_3d_data(path)
    sample_image[sample_image == -2000] = 0

    batch = []
    cnt = 0
    dx = 40
    ds = 512
    for i in range(0, sample_image.shape[0] - 3, 3):
        tmp = []
        for j in range(3):
            img = sample_image[i + j]
            img = 255.0 / np.amax(img) * img
            img = cv2.equalizeHist(img.astype(np.uint8))
            img = img[dx: ds - dx, dx: ds - dx]
            img = cv2.resize(img, (224, 224))
            tmp.append(img)

        tmp = np.array(tmp)
        batch.append(np.array(tmp))

    batch = np.array(batch)
    return batch
```


```python
def calc_features():
    net = get_conv_model()
    for i, folder in enumerate(glob.glob('stage1/*')):
        batch = get_scan(folder)
        feats = net.predict(batch)
        
        if i % 100 == 0:
            print("Processed {0} of {1}".format(i, n_patients))
            print("Shape of result:", feats.shape)
            print()

        np.save(folder.replace("stage1", "res_feats_stage1"), feats)

calc_features()
```

    [91m[Deprecation Warning] mxnet.model.FeedForward has been deprecated. Please use mxnet.mod.Module instead.[0m
    [91m[Deprecation Warning] mxnet.model.FeedForward has been deprecated. Please use mxnet.mod.Module instead.[0m
    Processed 0 of 1595
    Shape of result: (90, 2048)
    
    Processed 100 of 1595
    Shape of result: (61, 2048)
    
    Processed 200 of 1595
    Shape of result: (50, 2048)
    
    Processed 300 of 1595
    Shape of result: (74, 2048)
    
    Processed 400 of 1595
    Shape of result: (43, 2048)
    
    Processed 500 of 1595
    Shape of result: (42, 2048)
    
    Processed 600 of 1595
    Shape of result: (51, 2048)
    
    Processed 700 of 1595
    Shape of result: (77, 2048)
    
    Processed 800 of 1595
    Shape of result: (38, 2048)
    
    Processed 900 of 1595
    Shape of result: (45, 2048)
    
    Processed 1000 of 1595
    Shape of result: (73, 2048)
    
    Processed 1100 of 1595
    Shape of result: (38, 2048)
    
    Processed 1200 of 1595
    Shape of result: (47, 2048)
    
    Processed 1300 of 1595
    Shape of result: (93, 2048)
    
    Processed 1400 of 1595
    Shape of result: (47, 2048)
    
    Processed 1500 of 1595
    Shape of result: (55, 2048)
    
    


```python
def train_xgboost():
    df = pd.read_csv("metadata/stage1_labels.csv")

    x = np.array([np.mean(np.load('res_feats_stage1/%s.npy' % str(id)), axis=0) for id in df['id'].tolist()])
    y = df['cancer'].as_matrix()

    skf = StratifiedKFold(n_splits=5, random_state=2017, shuffle=True)

    clfs = []
    for train_index, test_index in skf.split(x, y):
        trn_x, val_x = x[train_index,:], x[test_index,:]
        trn_y, val_y = y[train_index], y[test_index]

        clf = xgb.XGBRegressor(max_depth=6,
                               n_estimators=1500,
                               min_child_weight=100,
                               learning_rate=0.037,
                               nthread=8,
                               subsample=0.9,
                               colsample_bytree=0.95,
                               seed=42)

        clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=50)
        clfs.append(clf)

    return clfs

# bst = train_xgboost()
```


```python
def make_submit():
    clfs = train_xgboost()

    df = pd.read_csv("submissions/stage1_sample_submission.csv")

    x = np.array([np.mean(np.load('res_feats_stage1/%s.npy' % str(id)), axis=0) for id in df['id'].values])
    
    preds = []
    for clf in clfs:
        preds.append(np.clip(clf.predict(x), 0.001, 1))

    pred = gmean(np.array(preds), axis=0)
    df['cancer'] = pred
    df.to_csv('submissions/-LB_xgb_0.037LR_1500EST_MXD6_5FOLD_E.csv', index=False)
    print(df.head())
    
make_submit()
```

    [0]	validation_0-logloss:0.685321
    Will train until validation_0-logloss hasn't improved in 50 rounds.
    [1]	validation_0-logloss:0.676795
    [2]	validation_0-logloss:0.669064
    [3]	validation_0-logloss:0.662653
    [4]	validation_0-logloss:0.656933
    [5]	validation_0-logloss:0.650806
    [6]	validation_0-logloss:0.64481
    [7]	validation_0-logloss:0.639115
    [8]	validation_0-logloss:0.635148
    [9]	validation_0-logloss:0.630157
    [10]	validation_0-logloss:0.626785
    [11]	validation_0-logloss:0.622782
    [12]	validation_0-logloss:0.618266
    [13]	validation_0-logloss:0.615927
    [14]	validation_0-logloss:0.611933
    [15]	validation_0-logloss:0.608504
    [16]	validation_0-logloss:0.605917
    [17]	validation_0-logloss:0.602814
    [18]	validation_0-logloss:0.600024
    [19]	validation_0-logloss:0.59809
    [20]	validation_0-logloss:0.596476
    [21]	validation_0-logloss:0.594396
    [22]	validation_0-logloss:0.591903
    [23]	validation_0-logloss:0.590782
    [24]	validation_0-logloss:0.589164
    [25]	validation_0-logloss:0.588074
    [26]	validation_0-logloss:0.586951
    [27]	validation_0-logloss:0.585649
    [28]	validation_0-logloss:0.585187
    [29]	validation_0-logloss:0.583326
    [30]	validation_0-logloss:0.582691
    [31]	validation_0-logloss:0.58163
    [32]	validation_0-logloss:0.581481
    [33]	validation_0-logloss:0.580794
    [34]	validation_0-logloss:0.580914
    [35]	validation_0-logloss:0.580855
    [36]	validation_0-logloss:0.580212
    [37]	validation_0-logloss:0.580753
    [38]	validation_0-logloss:0.580414
    [39]	validation_0-logloss:0.579428
    [40]	validation_0-logloss:0.579353
    [41]	validation_0-logloss:0.579222
    [42]	validation_0-logloss:0.578907
    [43]	validation_0-logloss:0.578721
    [44]	validation_0-logloss:0.578642
    [45]	validation_0-logloss:0.578796
    [46]	validation_0-logloss:0.578857
    [47]	validation_0-logloss:0.578772
    [48]	validation_0-logloss:0.578038
    [49]	validation_0-logloss:0.576707
    [50]	validation_0-logloss:0.576589
    [51]	validation_0-logloss:0.576105
    [52]	validation_0-logloss:0.57602
    [53]	validation_0-logloss:0.575833
    [54]	validation_0-logloss:0.576276
    [55]	validation_0-logloss:0.576267
    [56]	validation_0-logloss:0.576098
    [57]	validation_0-logloss:0.5755
    [58]	validation_0-logloss:0.575746
    [59]	validation_0-logloss:0.575849
    [60]	validation_0-logloss:0.575709
    [61]	validation_0-logloss:0.575516
    [62]	validation_0-logloss:0.574586
    [63]	validation_0-logloss:0.574562
    [64]	validation_0-logloss:0.574658
    [65]	validation_0-logloss:0.574543
    [66]	validation_0-logloss:0.57484
    [67]	validation_0-logloss:0.575018
    [68]	validation_0-logloss:0.574678
    [69]	validation_0-logloss:0.576169
    [70]	validation_0-logloss:0.575866
    [71]	validation_0-logloss:0.575467
    [72]	validation_0-logloss:0.574852
    [73]	validation_0-logloss:0.574789
    [74]	validation_0-logloss:0.574841
    [75]	validation_0-logloss:0.574893
    [76]	validation_0-logloss:0.575116
    [77]	validation_0-logloss:0.57598
    [78]	validation_0-logloss:0.576647
    [79]	validation_0-logloss:0.576925
    [80]	validation_0-logloss:0.577246
    [81]	validation_0-logloss:0.577038
    [82]	validation_0-logloss:0.577456
    [83]	validation_0-logloss:0.577616
    [84]	validation_0-logloss:0.57782
    [85]	validation_0-logloss:0.579056
    [86]	validation_0-logloss:0.579598
    [87]	validation_0-logloss:0.579902
    [88]	validation_0-logloss:0.580056
    [89]	validation_0-logloss:0.579416
    [90]	validation_0-logloss:0.580093
    [91]	validation_0-logloss:0.580682
    [92]	validation_0-logloss:0.579699
    [93]	validation_0-logloss:0.580215
    [94]	validation_0-logloss:0.580288
    [95]	validation_0-logloss:0.580791
    [96]	validation_0-logloss:0.581006
    [97]	validation_0-logloss:0.58027
    [98]	validation_0-logloss:0.580718
    [99]	validation_0-logloss:0.580706
    [100]	validation_0-logloss:0.58069
    [101]	validation_0-logloss:0.58105
    [102]	validation_0-logloss:0.580295
    [103]	validation_0-logloss:0.580041
    [104]	validation_0-logloss:0.580241
    [105]	validation_0-logloss:0.580617
    [106]	validation_0-logloss:0.58073
    [107]	validation_0-logloss:0.582329
    [108]	validation_0-logloss:0.582646
    [109]	validation_0-logloss:0.582762
    [110]	validation_0-logloss:0.584071
    [111]	validation_0-logloss:0.5845
    [112]	validation_0-logloss:0.586105
    [113]	validation_0-logloss:0.585561
    [114]	validation_0-logloss:0.586129
    [115]	validation_0-logloss:0.586185
    Stopping. Best iteration:
    [65]	validation_0-logloss:0.574543
    
    [0]	validation_0-logloss:0.685309
    Will train until validation_0-logloss hasn't improved in 50 rounds.
    [1]	validation_0-logloss:0.677826
    [2]	validation_0-logloss:0.669806
    [3]	validation_0-logloss:0.664132
    [4]	validation_0-logloss:0.658952
    [5]	validation_0-logloss:0.654009
    [6]	validation_0-logloss:0.648827
    [7]	validation_0-logloss:0.64392
    [8]	validation_0-logloss:0.638901
    [9]	validation_0-logloss:0.634481
    [10]	validation_0-logloss:0.631302
    [11]	validation_0-logloss:0.626619
    [12]	validation_0-logloss:0.623483
    [13]	validation_0-logloss:0.621415
    [14]	validation_0-logloss:0.617436
    [15]	validation_0-logloss:0.615298
    [16]	validation_0-logloss:0.612582
    [17]	validation_0-logloss:0.610178
    [18]	validation_0-logloss:0.60664
    [19]	validation_0-logloss:0.604884
    [20]	validation_0-logloss:0.60321
    [21]	validation_0-logloss:0.601814
    [22]	validation_0-logloss:0.601048
    [23]	validation_0-logloss:0.599924
    [24]	validation_0-logloss:0.598574
    [25]	validation_0-logloss:0.597006
    [26]	validation_0-logloss:0.596425
    [27]	validation_0-logloss:0.595371
    [28]	validation_0-logloss:0.59347
    [29]	validation_0-logloss:0.59248
    [30]	validation_0-logloss:0.59151
    [31]	validation_0-logloss:0.590271
    [32]	validation_0-logloss:0.59021
    [33]	validation_0-logloss:0.589515
    [34]	validation_0-logloss:0.589037
    [35]	validation_0-logloss:0.588463
    [36]	validation_0-logloss:0.587183
    [37]	validation_0-logloss:0.58723
    [38]	validation_0-logloss:0.586477
    [39]	validation_0-logloss:0.586206
    [40]	validation_0-logloss:0.585805
    [41]	validation_0-logloss:0.584846
    [42]	validation_0-logloss:0.585254
    [43]	validation_0-logloss:0.584666
    [44]	validation_0-logloss:0.584017
    [45]	validation_0-logloss:0.583558
    [46]	validation_0-logloss:0.583512
    [47]	validation_0-logloss:0.583177
    [48]	validation_0-logloss:0.581786
    [49]	validation_0-logloss:0.581947
    [50]	validation_0-logloss:0.582644
    [51]	validation_0-logloss:0.581762
    [52]	validation_0-logloss:0.581471
    [53]	validation_0-logloss:0.5813
    [54]	validation_0-logloss:0.581586
    [55]	validation_0-logloss:0.58195
    [56]	validation_0-logloss:0.581618
    [57]	validation_0-logloss:0.580626
    [58]	validation_0-logloss:0.581358
    [59]	validation_0-logloss:0.582034
    [60]	validation_0-logloss:0.581986
    [61]	validation_0-logloss:0.582594
    [62]	validation_0-logloss:0.582716
    [63]	validation_0-logloss:0.58323
    [64]	validation_0-logloss:0.583977
    [65]	validation_0-logloss:0.583942
    [66]	validation_0-logloss:0.583516
    [67]	validation_0-logloss:0.583332
    [68]	validation_0-logloss:0.583142
    [69]	validation_0-logloss:0.582569
    [70]	validation_0-logloss:0.582138
    [71]	validation_0-logloss:0.58201
    [72]	validation_0-logloss:0.582828
    [73]	validation_0-logloss:0.583748
    [74]	validation_0-logloss:0.582966
    [75]	validation_0-logloss:0.582964
    [76]	validation_0-logloss:0.583178
    [77]	validation_0-logloss:0.583777
    [78]	validation_0-logloss:0.583995
    [79]	validation_0-logloss:0.583426
    [80]	validation_0-logloss:0.583245
    [81]	validation_0-logloss:0.583549
    [82]	validation_0-logloss:0.583858
    [83]	validation_0-logloss:0.58389
    [84]	validation_0-logloss:0.584132
    [85]	validation_0-logloss:0.583683
    [86]	validation_0-logloss:0.584076
    [87]	validation_0-logloss:0.585091
    [88]	validation_0-logloss:0.585335
    [89]	validation_0-logloss:0.586161
    [90]	validation_0-logloss:0.586036
    [91]	validation_0-logloss:0.586735
    [92]	validation_0-logloss:0.587001
    [93]	validation_0-logloss:0.587473
    [94]	validation_0-logloss:0.587755
    [95]	validation_0-logloss:0.587494
    [96]	validation_0-logloss:0.588634
    [97]	validation_0-logloss:0.589415
    [98]	validation_0-logloss:0.589103
    [99]	validation_0-logloss:0.589181
    [100]	validation_0-logloss:0.588743
    [101]	validation_0-logloss:0.587889
    [102]	validation_0-logloss:0.587918
    [103]	validation_0-logloss:0.588039
    [104]	validation_0-logloss:0.587889
    [105]	validation_0-logloss:0.587915
    [106]	validation_0-logloss:0.588409
    [107]	validation_0-logloss:0.589212
    Stopping. Best iteration:
    [57]	validation_0-logloss:0.580626
    
    [0]	validation_0-logloss:0.6848
    Will train until validation_0-logloss hasn't improved in 50 rounds.
    [1]	validation_0-logloss:0.677382
    [2]	validation_0-logloss:0.670005
    [3]	validation_0-logloss:0.662193
    [4]	validation_0-logloss:0.655937
    [5]	validation_0-logloss:0.64916
    [6]	validation_0-logloss:0.644159
    [7]	validation_0-logloss:0.639797
    [8]	validation_0-logloss:0.635139
    [9]	validation_0-logloss:0.630144
    [10]	validation_0-logloss:0.625513
    [11]	validation_0-logloss:0.621587
    [12]	validation_0-logloss:0.618319
    [13]	validation_0-logloss:0.614458
    [14]	validation_0-logloss:0.611267
    [15]	validation_0-logloss:0.607429
    [16]	validation_0-logloss:0.60498
    [17]	validation_0-logloss:0.602542
    [18]	validation_0-logloss:0.599554
    [19]	validation_0-logloss:0.597564
    [20]	validation_0-logloss:0.594616
    [21]	validation_0-logloss:0.593282
    [22]	validation_0-logloss:0.591752
    [23]	validation_0-logloss:0.590808
    [24]	validation_0-logloss:0.58884
    [25]	validation_0-logloss:0.586679
    [26]	validation_0-logloss:0.584702
    [27]	validation_0-logloss:0.58392
    [28]	validation_0-logloss:0.582625
    [29]	validation_0-logloss:0.580539
    [30]	validation_0-logloss:0.579734
    [31]	validation_0-logloss:0.578482
    [32]	validation_0-logloss:0.577682
    [33]	validation_0-logloss:0.576701
    [34]	validation_0-logloss:0.575155
    [35]	validation_0-logloss:0.57444
    [36]	validation_0-logloss:0.573184
    [37]	validation_0-logloss:0.572142
    [38]	validation_0-logloss:0.572311
    [39]	validation_0-logloss:0.572458
    [40]	validation_0-logloss:0.571507
    [41]	validation_0-logloss:0.570888
    [42]	validation_0-logloss:0.570685
    [43]	validation_0-logloss:0.570152
    [44]	validation_0-logloss:0.569995
    [45]	validation_0-logloss:0.571233
    [46]	validation_0-logloss:0.570319
    [47]	validation_0-logloss:0.569938
    [48]	validation_0-logloss:0.569529
    [49]	validation_0-logloss:0.569472
    [50]	validation_0-logloss:0.568987
    [51]	validation_0-logloss:0.568899
    [52]	validation_0-logloss:0.569484
    [53]	validation_0-logloss:0.569155
    [54]	validation_0-logloss:0.568529
    [55]	validation_0-logloss:0.568452
    [56]	validation_0-logloss:0.568259
    [57]	validation_0-logloss:0.568104
    [58]	validation_0-logloss:0.568079
    [59]	validation_0-logloss:0.568157
    [60]	validation_0-logloss:0.568498
    [61]	validation_0-logloss:0.568513
    [62]	validation_0-logloss:0.569248
    [63]	validation_0-logloss:0.569089
    [64]	validation_0-logloss:0.568671
    [65]	validation_0-logloss:0.568788
    [66]	validation_0-logloss:0.568155
    [67]	validation_0-logloss:0.567874
    [68]	validation_0-logloss:0.567737
    [69]	validation_0-logloss:0.568094
    [70]	validation_0-logloss:0.567622
    [71]	validation_0-logloss:0.568419
    [72]	validation_0-logloss:0.569089
    [73]	validation_0-logloss:0.569282
    [74]	validation_0-logloss:0.569859
    [75]	validation_0-logloss:0.569995
    [76]	validation_0-logloss:0.570092
    [77]	validation_0-logloss:0.570034
    [78]	validation_0-logloss:0.570583
    [79]	validation_0-logloss:0.570449
    [80]	validation_0-logloss:0.569798
    [81]	validation_0-logloss:0.56945
    [82]	validation_0-logloss:0.569236
    [83]	validation_0-logloss:0.568829
    [84]	validation_0-logloss:0.569488
    [85]	validation_0-logloss:0.569951
    [86]	validation_0-logloss:0.570667
    [87]	validation_0-logloss:0.570177
    [88]	validation_0-logloss:0.570637
    [89]	validation_0-logloss:0.571662
    [90]	validation_0-logloss:0.570658
    [91]	validation_0-logloss:0.570596
    [92]	validation_0-logloss:0.570237
    [93]	validation_0-logloss:0.570862
    [94]	validation_0-logloss:0.570735
    [95]	validation_0-logloss:0.571096
    [96]	validation_0-logloss:0.571344
    [97]	validation_0-logloss:0.571187
    [98]	validation_0-logloss:0.571891
    [99]	validation_0-logloss:0.572441
    [100]	validation_0-logloss:0.571972
    [101]	validation_0-logloss:0.571955
    [102]	validation_0-logloss:0.572473
    [103]	validation_0-logloss:0.573109
    [104]	validation_0-logloss:0.573676
    [105]	validation_0-logloss:0.573613
    [106]	validation_0-logloss:0.574276
    [107]	validation_0-logloss:0.574297
    [108]	validation_0-logloss:0.574436
    [109]	validation_0-logloss:0.574473
    [110]	validation_0-logloss:0.573602
    [111]	validation_0-logloss:0.574021
    [112]	validation_0-logloss:0.573708
    [113]	validation_0-logloss:0.574098
    [114]	validation_0-logloss:0.574237
    [115]	validation_0-logloss:0.57454
    [116]	validation_0-logloss:0.574828
    [117]	validation_0-logloss:0.575426
    [118]	validation_0-logloss:0.575185
    [119]	validation_0-logloss:0.575868
    [120]	validation_0-logloss:0.5761
    Stopping. Best iteration:
    [70]	validation_0-logloss:0.567622
    
    [0]	validation_0-logloss:0.682986
    Will train until validation_0-logloss hasn't improved in 50 rounds.
    [1]	validation_0-logloss:0.675607
    [2]	validation_0-logloss:0.668248
    [3]	validation_0-logloss:0.661721
    [4]	validation_0-logloss:0.655316
    [5]	validation_0-logloss:0.649052
    [6]	validation_0-logloss:0.643335
    [7]	validation_0-logloss:0.637996
    [8]	validation_0-logloss:0.632953
    [9]	validation_0-logloss:0.628322
    [10]	validation_0-logloss:0.622889
    [11]	validation_0-logloss:0.619497
    [12]	validation_0-logloss:0.615277
    [13]	validation_0-logloss:0.611624
    [14]	validation_0-logloss:0.608109
    [15]	validation_0-logloss:0.604556
    [16]	validation_0-logloss:0.601946
    [17]	validation_0-logloss:0.598724
    [18]	validation_0-logloss:0.596145
    [19]	validation_0-logloss:0.593473
    [20]	validation_0-logloss:0.591949
    [21]	validation_0-logloss:0.590707
    [22]	validation_0-logloss:0.588731
    [23]	validation_0-logloss:0.586629
    [24]	validation_0-logloss:0.584885
    [25]	validation_0-logloss:0.584854
    [26]	validation_0-logloss:0.583665
    [27]	validation_0-logloss:0.582947
    [28]	validation_0-logloss:0.582042
    [29]	validation_0-logloss:0.580879
    [30]	validation_0-logloss:0.58002
    [31]	validation_0-logloss:0.578568
    [32]	validation_0-logloss:0.577465
    [33]	validation_0-logloss:0.577259
    [34]	validation_0-logloss:0.577382
    [35]	validation_0-logloss:0.576496
    [36]	validation_0-logloss:0.576768
    [37]	validation_0-logloss:0.576298
    [38]	validation_0-logloss:0.575908
    [39]	validation_0-logloss:0.575598
    [40]	validation_0-logloss:0.575848
    [41]	validation_0-logloss:0.574714
    [42]	validation_0-logloss:0.575084
    [43]	validation_0-logloss:0.574565
    [44]	validation_0-logloss:0.573869
    [45]	validation_0-logloss:0.573035
    [46]	validation_0-logloss:0.572675
    [47]	validation_0-logloss:0.573099
    [48]	validation_0-logloss:0.573174
    [49]	validation_0-logloss:0.572467
    [50]	validation_0-logloss:0.572381
    [51]	validation_0-logloss:0.571651
    [52]	validation_0-logloss:0.571316
    [53]	validation_0-logloss:0.572074
    [54]	validation_0-logloss:0.572104
    [55]	validation_0-logloss:0.571403
    [56]	validation_0-logloss:0.571669
    [57]	validation_0-logloss:0.571852
    [58]	validation_0-logloss:0.572084
    [59]	validation_0-logloss:0.571357
    [60]	validation_0-logloss:0.571139
    [61]	validation_0-logloss:0.571923
    [62]	validation_0-logloss:0.572075
    [63]	validation_0-logloss:0.572906
    [64]	validation_0-logloss:0.573612
    [65]	validation_0-logloss:0.573821
    [66]	validation_0-logloss:0.574124
    [67]	validation_0-logloss:0.574236
    [68]	validation_0-logloss:0.573827
    [69]	validation_0-logloss:0.574023
    [70]	validation_0-logloss:0.574812
    [71]	validation_0-logloss:0.575686
    [72]	validation_0-logloss:0.575531
    [73]	validation_0-logloss:0.576167
    [74]	validation_0-logloss:0.576144
    [75]	validation_0-logloss:0.575932
    [76]	validation_0-logloss:0.576619
    [77]	validation_0-logloss:0.576575
    [78]	validation_0-logloss:0.57798
    [79]	validation_0-logloss:0.578123
    [80]	validation_0-logloss:0.578478
    [81]	validation_0-logloss:0.578932
    [82]	validation_0-logloss:0.578922
    [83]	validation_0-logloss:0.579131
    [84]	validation_0-logloss:0.580165
    [85]	validation_0-logloss:0.580693
    [86]	validation_0-logloss:0.58116
    [87]	validation_0-logloss:0.581597
    [88]	validation_0-logloss:0.582369
    [89]	validation_0-logloss:0.58271
    [90]	validation_0-logloss:0.584608
    [91]	validation_0-logloss:0.584953
    [92]	validation_0-logloss:0.584589
    [93]	validation_0-logloss:0.586201
    [94]	validation_0-logloss:0.589441
    [95]	validation_0-logloss:0.5918
    [96]	validation_0-logloss:0.587921
    [97]	validation_0-logloss:0.588097
    [98]	validation_0-logloss:0.590414
    [99]	validation_0-logloss:0.597984
    [100]	validation_0-logloss:0.701581
    [101]	validation_0-logloss:0.594218
    [102]	validation_0-logloss:0.587387
    [103]	validation_0-logloss:0.589883
    [104]	validation_0-logloss:0.593328
    [105]	validation_0-logloss:0.589957
    [106]	validation_0-logloss:0.591506
    [107]	validation_0-logloss:0.594912
    [108]	validation_0-logloss:0.703236
    [109]	validation_0-logloss:0.703552
    [110]	validation_0-logloss:0.703459
    Stopping. Best iteration:
    [60]	validation_0-logloss:0.571139
    
    [0]	validation_0-logloss:0.684221
    Will train until validation_0-logloss hasn't improved in 50 rounds.
    [1]	validation_0-logloss:0.676145
    [2]	validation_0-logloss:0.669139
    [3]	validation_0-logloss:0.662061
    [4]	validation_0-logloss:0.655997
    [5]	validation_0-logloss:0.650029
    [6]	validation_0-logloss:0.642697
    [7]	validation_0-logloss:0.637253
    [8]	validation_0-logloss:0.632742
    [9]	validation_0-logloss:0.627865
    [10]	validation_0-logloss:0.623806
    [11]	validation_0-logloss:0.62058
    [12]	validation_0-logloss:0.616422
    [13]	validation_0-logloss:0.61239
    [14]	validation_0-logloss:0.609222
    [15]	validation_0-logloss:0.606138
    [16]	validation_0-logloss:0.602992
    [17]	validation_0-logloss:0.599781
    [18]	validation_0-logloss:0.598059
    [19]	validation_0-logloss:0.5956
    [20]	validation_0-logloss:0.593631
    [21]	validation_0-logloss:0.591011
    [22]	validation_0-logloss:0.589679
    [23]	validation_0-logloss:0.587474
    [24]	validation_0-logloss:0.584753
    [25]	validation_0-logloss:0.583256
    [26]	validation_0-logloss:0.581633
    [27]	validation_0-logloss:0.580804
    [28]	validation_0-logloss:0.578896
    [29]	validation_0-logloss:0.577717
    [30]	validation_0-logloss:0.57791
    [31]	validation_0-logloss:0.575698
    [32]	validation_0-logloss:0.574096
    [33]	validation_0-logloss:0.573211
    [34]	validation_0-logloss:0.572278
    [35]	validation_0-logloss:0.571236
    [36]	validation_0-logloss:0.570665
    [37]	validation_0-logloss:0.569974
    [38]	validation_0-logloss:0.569058
    [39]	validation_0-logloss:0.567815
    [40]	validation_0-logloss:0.567264
    [41]	validation_0-logloss:0.565923
    [42]	validation_0-logloss:0.565592
    [43]	validation_0-logloss:0.566248
    [44]	validation_0-logloss:0.565422
    [45]	validation_0-logloss:0.565339
    [46]	validation_0-logloss:0.565349
    [47]	validation_0-logloss:0.565157
    [48]	validation_0-logloss:0.56481
    [49]	validation_0-logloss:0.564364
    [50]	validation_0-logloss:0.564198
    [51]	validation_0-logloss:0.56393
    [52]	validation_0-logloss:0.563829
    [53]	validation_0-logloss:0.563513
    [54]	validation_0-logloss:0.563573
    [55]	validation_0-logloss:0.564609
    [56]	validation_0-logloss:0.563343
    [57]	validation_0-logloss:0.562488
    [58]	validation_0-logloss:0.562444
    [59]	validation_0-logloss:0.562806
    [60]	validation_0-logloss:0.562446
    [61]	validation_0-logloss:0.562809
    [62]	validation_0-logloss:0.563409
    [63]	validation_0-logloss:0.562674
    [64]	validation_0-logloss:0.561812
    [65]	validation_0-logloss:0.562224
    [66]	validation_0-logloss:0.561778
    [67]	validation_0-logloss:0.561295
    [68]	validation_0-logloss:0.560841
    [69]	validation_0-logloss:0.560499
    [70]	validation_0-logloss:0.560494
    [71]	validation_0-logloss:0.560457
    [72]	validation_0-logloss:0.561025
    [73]	validation_0-logloss:0.560668
    [74]	validation_0-logloss:0.560982
    [75]	validation_0-logloss:0.561653
    [76]	validation_0-logloss:0.561926
    [77]	validation_0-logloss:0.561608
    [78]	validation_0-logloss:0.561201
    [79]	validation_0-logloss:0.560907
    [80]	validation_0-logloss:0.56037
    [81]	validation_0-logloss:0.560077
    [82]	validation_0-logloss:0.559243
    [83]	validation_0-logloss:0.559444
    [84]	validation_0-logloss:0.55943
    [85]	validation_0-logloss:0.558959
    [86]	validation_0-logloss:0.559264
    [87]	validation_0-logloss:0.5586
    [88]	validation_0-logloss:0.558784
    [89]	validation_0-logloss:0.558882
    [90]	validation_0-logloss:0.559875
    [91]	validation_0-logloss:0.559619
    [92]	validation_0-logloss:0.559222
    [93]	validation_0-logloss:0.558407
    [94]	validation_0-logloss:0.558439
    [95]	validation_0-logloss:0.558841
    [96]	validation_0-logloss:0.559205
    [97]	validation_0-logloss:0.559887
    [98]	validation_0-logloss:0.559629
    [99]	validation_0-logloss:0.560525
    [100]	validation_0-logloss:0.561786
    [101]	validation_0-logloss:0.561021
    [102]	validation_0-logloss:0.562152
    [103]	validation_0-logloss:0.562519
    [104]	validation_0-logloss:0.5629
    [105]	validation_0-logloss:0.56286
    [106]	validation_0-logloss:0.562345
    [107]	validation_0-logloss:0.56271
    [108]	validation_0-logloss:0.562808
    [109]	validation_0-logloss:0.562807
    [110]	validation_0-logloss:0.562939
    [111]	validation_0-logloss:0.56378
    [112]	validation_0-logloss:0.564236
    [113]	validation_0-logloss:0.565075
    [114]	validation_0-logloss:0.56594
    [115]	validation_0-logloss:0.565808
    [116]	validation_0-logloss:0.565831
    [117]	validation_0-logloss:0.564993
    [118]	validation_0-logloss:0.564795
    [119]	validation_0-logloss:0.565168
    [120]	validation_0-logloss:0.565441
    [121]	validation_0-logloss:0.565726
    [122]	validation_0-logloss:0.565146
    [123]	validation_0-logloss:0.565907
    [124]	validation_0-logloss:0.565636
    [125]	validation_0-logloss:0.565453
    [126]	validation_0-logloss:0.564626
    [127]	validation_0-logloss:0.564241
    [128]	validation_0-logloss:0.5656
    [129]	validation_0-logloss:0.565951
    [130]	validation_0-logloss:0.565769
    [131]	validation_0-logloss:0.565517
    [132]	validation_0-logloss:0.565498
    [133]	validation_0-logloss:0.565236
    [134]	validation_0-logloss:0.565309
    [135]	validation_0-logloss:0.56506
    [136]	validation_0-logloss:0.564617
    [137]	validation_0-logloss:0.565472
    [138]	validation_0-logloss:0.566067
    [139]	validation_0-logloss:0.566185
    [140]	validation_0-logloss:0.566276
    [141]	validation_0-logloss:0.566304
    [142]	validation_0-logloss:0.565639
    [143]	validation_0-logloss:0.56544
    Stopping. Best iteration:
    [93]	validation_0-logloss:0.558407
    
                                     id    cancer
    0  026470d51482c93efc18b9803159c960  0.410584
    1  031b7ec4fe96a3b035a8196264a8c8c3  0.356920
    2  03bd22ed5858039af223c04993e9eb22  0.462064
    3  06a90409e4fcea3e634748b967993531  0.334250
    4  07b1defcfae5873ee1f03c90255eb170  0.318110
    
