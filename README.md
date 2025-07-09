# 🧠 PCOS Ultrasound Image Classification using Two-Stream CNN & Transformer Attention

This project focuses on detecting signs of **Polycystic Ovary Syndrome (PCOS)** from **ultrasound images** of ovaries using a deep learning model that combines:

- A **Two-Stream Convolutional Neural Network (CNN)** to analyze the upper and lower halves of the image separately.
- A **Transformer-based Multi-Head Attention** mechanism to understand complex interactions in features.

The model classifies each ultrasound image into either:

- ✅ `noninfected` — healthy ovaries  
- 🚫 `infected` — ovaries with PCOS

---

## 📦 1. Dataset

We use the [**PCOS-XAI-Ultrasound Dataset**](https://www.kaggle.com/datasets/ibadeus/pcos-xai-ultrasound-dataset), which contains labeled ultrasound images of ovaries.

- **infected/**: Images of ovaries diagnosed with PCOS  
- **noninfected/**: Normal, healthy ovary images  

You can download it using:

```python
import kagglehub

path = kagglehub.dataset_download("ibadeus/pcos-xai-ultrasound-dataset")
```

---

## 🧹 2. Data Preprocessing & Augmentation

- Encoded labels using `LabelEncoder`
- Resampled minority class to **balance dataset**
- Applied image rescaling with `ImageDataGenerator`

```python
from sklearn.utils import resample

max_count = df['category_encoded'].value_counts().max()
dfs = []

for category in df['category_encoded'].unique():
    class_subset = df[df['category_encoded'] == category]
    class_upsampled = resample(class_subset, replace=True, n_samples=max_count, random_state=42)
    dfs.append(class_upsampled)

df_balanced = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
```

---

## 🧠 3. Model Architecture

The architecture combines:

- Two CNN branches:
  - One for the upper half of the image
  - One for the (flipped) lower half
- Dense layers for feature embedding
- Transformer’s **Multi-Head Attention** to merge features
- Final classification using a `Dense` softmax layer

🖼️ **Architecture Overview**:

![Model Architecture](model.png)

---

## 🧪 4. Evaluation

The model was trained with:

- `sparse_categorical_crossentropy` loss
- `Adam` optimizer
- Accuracy metrics

📊 **Classification Report:**

```
              precision    recall  f1-score   support

           0       1.00      0.98      0.99       679
           1       0.98      1.00      0.99       678

    accuracy                           0.99      1357
   macro avg       0.99      0.99      0.99      1357
weighted avg       0.99      0.99      0.99      1357
```

✅ **Test Accuracy:** `99%`

---

## 💾 5. Model Files

- `model_architecture.json`: JSON representation of the model
- `model_weights.weights.h5`: Saved weights of the model
- `model.png`: Visual plot of the model architecture

You can load the model with:

```python
from tensorflow.keras.models import model_from_json

with open("model_architecture.json", "r") as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json, custom_objects={
    "split_image": split_image,
    "flip_lower_half": flip_lower_half
})
model.load_weights("model_weights.weights.h5")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

---

## 📁 Project Structure

```bash
PCOS-Ultrasound-Classification/
├── README.md                      # This documentation file
├── model.png                      # Model architecture plot
├── model_architecture.json        # Model structure in JSON
├── model_weights.weights.h5       # Trained weights
├── requirements.txt               # Python dependencies
├── pcos_classification.ipynb      # Main Jupyter Notebook
```

---

## 📚 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Future Work

- Deploy model as a web API using Flask or FastAPI
- Integrate Grad-CAM or SHAP for explainability
- Extend to multi-class ovarian disorders
