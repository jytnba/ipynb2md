

```python
import os
import numpy as np
import pandas as pd

from glob import glob
```


```python
label_df = pd.read_csv('metadata/stage1_labels.csv')

label_df.head()
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
      <td>0015ceb851d7251b8f399e39779d1e7d</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0030a160d58723ff36d73f41b170ec21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>003f41c78e6acfa92430a057ac0b306e</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>006b96310a37b36cccb2ab48d10b49a3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>008464bb8521d09a42985dd8add3d0d2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
%cd prepd_stage1/
```

    /home/robert/projects/kaggle/cancer/prepd_stage1
    


```python
g = glob('*.npy')
patients = [patient.replace(".npy", "") for patient in g]

patients[:10]
```




    ['968808e60982c76366088b4db7254671',
     'caddd8f856b110ed9bb52872e003193f',
     'fbae4d04285789dfa32124c86586dd09',
     '6997b392b28529c9ab67add45055fcf7',
     '645e7f46eb9b834153ecf8e2b2921fe5',
     'a4ae73433e63549558a3d0eed9128a69',
     '9b4f0020e407fd7f980f4241fb6ac7ce',
     '487cc9003c99fa95e9a9e201d396992c',
     '7b4d476d9414c9339269efe47840d2da',
     '310b403db75226a9a44d9a29c18917b7']




```python
test_ids = [patient for patient in patients if patient not in label_df["id"].values]

print(len(test_ids))
test_ids[:5]
```

    198
    




    ['5d16819bd78c74448ce852a93bf423ad',
     'a6c15206edadab0270898f03e770d730',
     'e6d8ae8c3b0817df994a1ce3b37a7efb',
     '70f4eb8201e3155cc3e399f0ff09c5ef',
     '7daeb8ef7307849c715f7f6f3e2dd88e']




```python
for id in test_ids:
    fn = "{}.npy".format(id)
    os.rename(fn, './test/' + fn)
```


```python
g = glob('*.npy')
shuf = np.random.permutation(g)

for i in range(200):
    os.rename(shuf[i], './valid/' + shuf[i])
```
