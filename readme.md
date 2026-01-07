# PlantVillage Dataset – Week 1 EDA Report

## Objective

Perform exploratory data analysis (EDA) on the PlantVillage dataset to understand class distribution, image properties, and differences between color, grayscale, and segmented images. This analysis prepares the dataset for future ML/CNN modeling.

---

## Dataset Overview

The PlantVillage dataset contains leaf images labeled by plant type and disease. Three variants are analyzed:

* **Color images (RGB)**
* **Grayscale images**
* **Segmented images** (background removed)

Each class corresponds to a specific plant–disease combination.

---

## Data Organization

* Images are stored in class-wise folders
* Folder names are used as class labels
* Separate DataFrames are created for color, grayscale, and segmented datasets

---

## Key Analyses & Observations

### 1. Class Distribution

* Dataset is **class-imbalanced**
* Some disease classes have significantly more samples
* Imbalance must be handled during training

### 2. Image Dimensions & Aspect Ratio

* Most images have similar sizes
* Minor variations exist → resizing required
* Aspect ratios are largely consistent with few outliers

### 3. Visual Inspection

* Sample images confirm clear disease patterns
* Segmented images provide cleaner leaf focus

### 4. Color Intensity (RGB)

* Green channel dominates (expected for leaf images)
* RGB intensity distributions vary across classes

---

## Tools Used

NumPy, Pandas, Matplotlib, Seaborn, Plotly, OpenCV, Dask

---

## Key Takeaways

* Dataset is large but imbalanced
* Preprocessing (resize, normalize) is necessary
* Segmented images are likely beneficial for modeling

---

## Next Steps

* Data preprocessing
* Handle class imbalance
* Train baseline CNN models
* Compare performance across dataset variants

---

**Conclusion:**
This EDA provides essential insights into the PlantVillage dataset and informs preprocessing and modeling decisions for subsequent stages.



# PlantVillage Dataset – Week 2 Report

## Objective

Establish a **baseline machine learning model** for plant disease classification using the PlantVillage dataset. This week focuses only on a **Random Forest classifier** to evaluate how well classical ML performs on image data.

---

## Dataset & Preprocessing

* Dataset variants: **Color / Grayscale / Segmented images**
* Images resized to a fixed resolution
* Pixel values normalized
* Images flattened into 1D feature vectors
* Labels encoded numerically
* Train–validation split applied

---

## Model Used

### Random Forest Classifier

* Ensemble-based machine learning model
* Uses multiple decision trees with bagging
* Serves as a **non-deep-learning baseline**

---

## Training & Evaluation

* Model trained on flattened image features
* Performance evaluated using validation accuracy

### Observations:

* Model learns basic patterns but struggles with complex spatial features
* Performance is limited due to loss of spatial information during flattening

---

## Results Summary

* Random Forest provides a reasonable baseline
* Performance is significantly lower than expected for image tasks
* Highlights the limitation of classical ML for high-dimensional image data

---

## Key Takeaways

* Random Forest is not ideal for raw image classification
* Spatial feature extraction is crucial
* Deep learning models are needed for better performance

---

## Next Steps

* Move to CNN-based models
* Avoid flattening images
* Use convolutional layers for feature extraction

---

**Conclusion:**
Week 2 demonstrates the limitations of classical machine learning models like Random Forest for image-based plant disease classification and motivates the transition to deep learning approaches.

# PlantVillage Dataset – Week 3 Report

## Objective

Move beyond classical machine learning and build **deep learning models that learn features directly from images**. This week focuses on training a **CNN from scratch** and then improving performance using **transfer learning**.

---

## Why Deep Learning

* Classical ML requires manual feature extraction
* Flattening images destroys spatial information
* CNNs automatically learn:

  * edges
  * textures
  * disease patterns
* Spatial structure is preserved, leading to better performance

---

## Approach

### Stage 1: CNN From Scratch

* Convolution → ReLU → MaxPooling blocks
* Fully connected classification head
* Trained for limited epochs
* **Purpose:** understand CNN basics and overfitting
* **Expected Accuracy:** ~70–80%

### Stage 2: Transfer Learning

* Pretrained backbone (e.g., MobileNet / ResNet)
* Backbone layers frozen
* Custom classifier head added
* Optional fine-tuning of last layers
* **Expected Accuracy:** ~85–90%+

---

## Preprocessing

* Resize images to 224×224
* Normalize pixel values
* Apply data augmentation (flip, rotate, zoom)

---

## Evaluation

* Accuracy comparison with Week 2 (Random Forest)
* Confusion matrix
* Training vs validation curves

---

## Key Observations

* CNN significantly outperforms classical ML
* Transfer learning converges faster and generalizes better
* Pretrained features are highly effective for plant disease detection

---

## Takeaways

* Deep learning eliminates manual feature engineering
* Spatial feature learning is critical for image tasks
* Transfer learning provides large accuracy gains with fewer epochs

---

**Conclusion:**
Week 3 demonstrates the real power of deep learning by showing how CNNs and transfer learning drastically outperform classical machine learning models on image-based disease classification.

