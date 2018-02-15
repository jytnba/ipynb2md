
# Data Science Bowl: Detecting Lung Cancer From Chest Scans


```python
import numpy as np
import pandas as pd
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

%matplotlib notebook
# Some constants 
INPUT_FOLDER = 'samples/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()
```

## Loading the files

These files contain a lot of metadata (such as the pixel size, so how long one pixel is in every dimension in the real world).

The pixel size/coarseness of the scan differs from scan to scan (e.g. the distance between slices may differ), which can hurt performance of CNN models. We can deal with this by isomorphic resampling, which we will do later.

Below is code to load a scan, which consists of multiple slices, which we simply save in a Python list. Every folder in the dataset is one scan (so one patient). One metadata field is missing, the pixel size in the Z direction, which is the slice thickness. Fortunately we can infer this, and we add this to the metadata.


```python
# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    
    slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices
```

The unit of measurement in CT scans is the Hounsfield Unit (HU), which is a measure of radiodensity. CT scanners are carefully calibrated to accurately measure this. By default however, the returned values are not in this unit.

Some scanners have cylindrical scanning bounds, but the output image is square. The pixels that fall outside of these bounds get the fixed value -2000. The first step is setting these values to 0, which currently corresponds to air. Next, convert back to HU units, by adding the intercept (which is stored in the metadata of the scans).


```python
def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
      
    intercept = scans[0].RescaleIntercept
    image += int(intercept)
    
    return np.array(image)
```

Example patient: 


```python
first_patient = load_scan(INPUT_FOLDER + patients[0])
first_patient_pixels = get_pixels_hu(first_patient)
plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()

# Show some slice in the middle
plt.imshow(first_patient_pixels[80], cmap=plt.cm.gray)
plt.show()
```


![png](output_8_0.png)



![png](output_8_1.png)


## Resampling

A scan may have a pixel spacing of [2.5, 0.5, 0.5], which means that the distance between slices is 2.5 millimeters. For a different scan this may be [1.5, 0.725, 0.725], this can be problematic for automatic analysis (e.g. using ConvNets).

A common method of dealing with this is resampling the full dataset to a certain isotropic resolution. If we choose to resample everything to 1mm1mm1mm pixels we can use 3D convnets without worrying about learning zoom/slice thickness invariance.


```python
def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, new_spacing
```

Resampling on example patient:


```python
pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])
print("Shape before resampling\t", first_patient_pixels.shape)
print("Shape after resampling\t", pix_resampled.shape)
```

    Shape before resampling	 (134, 512, 512)
    Shape after resampling	 (335, 306, 306)
    

## 3D plotting the scan

This takes a threshold argument which we can use to plot certain structures, such as all tissue or only the bones. 400 is a good threshold for showing the bones only.


```python
def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    p = p[:,:,::-1]
    
    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")

    ax.set_xlim(0, p.shape[0])  # a = 6 (times two for 2nd ellipsoid)
    ax.set_ylim(0, p.shape[1])  # b = 10
    ax.set_zlim(0, p.shape[2])  # c = 16

    plt.show()
```


```python
plot_3d(pix_resampled, 400)
```


![png](output_17_0.png)


## Lung segmentation

In order to reduce the problem space, segment the lungs (and usually some tissue around it). This consists of a series of applications of region growing and morphological operations. In this case, we will use only connected component analysis.

The steps:
- Threshold the image (-300 HU is a good threshold, but it doesn't matter much for this approach)
- Do connected components, determine label of air around person, fill this with 1s in the binary image
- Optionally: For every axial slice in the scan, determine the largest solid connected component (the body+air around the person), and set others to 0. This fills the structures in the lungs in the mask.
- Keep only the largest air pocket (the human body has other pockets of air here and there).


```python
def largest_label_volume(im, bg=-1):
    label_values = list(np.unique(im))
    if bg in label_values:
        label_values.remove(bg)
    biggest = label_values[np.argmax([np.sum(im == l) for l in label_values])]
    return biggest


def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -300, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    # Fill the air around the person
    binary_image[background_label == labels] = 2
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            binary_image[i][labeling != l_max] = 1

        
    binary_image -= 1 # Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l = largest_label_volume(labels, bg=0)
    binary_image[labels != l] = 0
 
    return binary_image
```


```python
segmented_lungs = segment_lung_mask(pix_resampled, False)
segmented_lungs_fill = segment_lung_mask(pix_resampled, True)

plot_3d(segmented_lungs, 0)
```


![png](output_21_0.png)


It's probably a good idea to include structures within the lung (as the nodules are solid), we do not only want to air in the lungs.


```python
plot_3d(segmented_lungs_fill, 0)
```


![png](output_23_0.png)


Visualizing the difference between the two:


```python
plot_3d(segmented_lungs_fill - segmented_lungs, 0)
```


![png](output_25_0.png)

