

```python
import numpy as np
import pandas as pd
import os
```


```python
num_subs = 2.0
id_col = 'id'
base_path = 'submissions/'
```


```python
dfs = []

df1 = pd.read_csv(base_path + 'xgb/0.576-LB_xgb_.01LR_1750EST_MCD10_A.csv')
dfs.append(df1)

df2 = pd.read_csv(base_path + 'xgb/0.599-LB_xgb_.03LR_1650EST_MXD9_B.csv')
dfs.append(df2)

# df3 = pd.read_csv(base_path + '0.058-LB_264x264_5epoch_3aug_0.001lr_0.5dropout_0.01clip_3voters_A.csv')
# dfs.append(df3)

# df4 = pd.read_csv(base_path + '0.065-LB_300x300_6epoch_3aug_0.001lr_0.5dropout_0.01clip_1voters_C.csv')
# dfs.append(df4)

# df5 = pd.read_csv(base_path + '.csv')
# dfs.append(df5)

# df6 = pd.read_csv(base_path + '.csv')
# dfs.append(df6)

df1.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>cancer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>026470d51482c93efc18b9803159c960</td>
      <td>0.228542</td>
    </tr>
    <tr>
      <th>1</th>
      <td>031b7ec4fe96a3b035a8196264a8c8c3</td>
      <td>0.316266</td>
    </tr>
    <tr>
      <th>2</th>
      <td>03bd22ed5858039af223c04993e9eb22</td>
      <td>0.243177</td>
    </tr>
    <tr>
      <th>3</th>
      <td>06a90409e4fcea3e634748b967993531</td>
      <td>0.380493</td>
    </tr>
    <tr>
      <th>4</th>
      <td>07b1defcfae5873ee1f03c90255eb170</td>
      <td>0.282819</td>
    </tr>
  </tbody>
</table>
</div>




```python
def create_ave_df(df_list):
    sample = df_list[0]
    ave_df = pd.DataFrame(columns=sample.columns)
    ave_df[id_col] = sample[id_col]
    ave_df.fillna(0.0, inplace=True)
    return ave_df

ave_df = create_ave_df(dfs)

ave_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>cancer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>026470d51482c93efc18b9803159c960</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>031b7ec4fe96a3b035a8196264a8c8c3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>03bd22ed5858039af223c04993e9eb22</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>06a90409e4fcea3e634748b967993531</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>07b1defcfae5873ee1f03c90255eb170</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
def populate_ave_df(ave_df, df_list):
    for col in ave_df.columns.values:
        if col == id_col:
            continue
        for df in df_list:
            ave_df[col] = ave_df[col] + df[col]
        ave_df[col] = ave_df[col] / num_subs
    return ave_df

ave_df = populate_ave_df(ave_df, dfs)
ave_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>cancer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>026470d51482c93efc18b9803159c960</td>
      <td>0.167498</td>
    </tr>
    <tr>
      <th>1</th>
      <td>031b7ec4fe96a3b035a8196264a8c8c3</td>
      <td>0.309935</td>
    </tr>
    <tr>
      <th>2</th>
      <td>03bd22ed5858039af223c04993e9eb22</td>
      <td>0.241118</td>
    </tr>
    <tr>
      <th>3</th>
      <td>06a90409e4fcea3e634748b967993531</td>
      <td>0.326073</td>
    </tr>
    <tr>
      <th>4</th>
      <td>07b1defcfae5873ee1f03c90255eb170</td>
      <td>0.317292</td>
    </tr>
  </tbody>
</table>
</div>




```python
def generate_new_submission(ave_df):
    ave_df.to_csv(base_path + '/-LB_.csv', index=False)
    
generate_new_submission(ave_df)
```

## Calculating correlation between submissions


```python
# dfs = []

# for root, dirs, files in os.walk(base_path):
#     for fn in [os.path.join(root, name) for name in files]:
# #         print fn
#         df = pd.read_csv(fn)
#         dfs.append(df)
        
# print len(dfs)
```


```python
### using sequential comparisons

# filtered_dfs = []

# for i in range(len(dfs)-1):
#     saved_ids = dfs[i][id_col]
#     dfa = dfs[i].drop(id_col, axis=1)
#     dfb = dfs[i+1].drop(id_col, axis=1)
#     corr = dfa.corrwith(dfb).sum()/dfa.shape[1]
# #     print corr
    
#     if corr < .7:
#         dfa[id_col] = saved_ids
#         filtered_dfs.append(dfa)

# num_subs = len(filtered_dfs)
# print "number meeting criteria:", num_subs
```


```python
### using averages

# filtered_dfs = []

# for dfa in dfs:
#     saved_ids = dfa[id_col]
#     dfa = dfa.drop(id_col, axis=1)
#     corr_sum = 0.
#     for dfb in dfs:
#         dfb = dfb.drop(id_col, axis=1)
#         corr = dfa.corrwith(dfb).sum()/dfa.shape[1]
#         corr_sum += corr
#     corr_avg = corr_sum/len(dfs)
# #     print corr_avg
    
#     if corr_avg < .95:
#         dfa[id_col] = saved_ids
#         filtered_dfs.append(dfa)

# num_subs = len(filtered_dfs)
# print "number meeting criteria:", num_subs
```


```python
# ave_df = create_ave_df(filtered_dfs)
# ave_df = populate_ave_df(ave_df, filtered_dfs)

# ave_df.head()
```


```python
# generate_new_submission(ave_df)
```
