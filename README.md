# PCOS Ultrasound Image Classification using Two-Stream CNN & Transformer Attention

"PCOS Ultrasound Image Classification using Two-Stream CNN & Transformer Attention"

This project focuses on detecting signs of Polycystic Ovary Syndrome (PCOS) from ultrasound images of ovaries. To achieve this, a custom deep learning model was designed that:

Splits each image into two parts (top and bottom halves) to capture distinct patterns more effectively.
Processes each half through separate CNN branches (Two-Stream CNN).
Combines the extracted features using a Transformer-based Multi-Head Attention mechanism to learn complex relationships.

The model is trained to classify each ultrasound image as either "PCOS infected" or "non-infected", helping in early diagnosis with high accuracy.


## 🔍 3. Dataset Details


## 📂 Dataset

The dataset used is **[PCOS-XAI-Ultrasound](https://www.kaggle.com/datasets/ibadeus/pcos-xai-ultrasound-dataset)** containing ultrasound images categorized as:

- `infected` (with PCOS)
- `noninfected`

It was downloaded using:

```python
import kagglehub
path = kagglehub.dataset_download("ibadeus/pcos-xai-ultrasound-dataset")

---

## 🧹 4. Data Preprocessing & Augmentation

```markdown
## 🧹 Data Preprocessing & Augmentation

Balanced and augmented the dataset to handle class imbalance:

```python
from sklearn.utils import resample

# Balance dataset
max_count = df['category_encoded'].value_counts().max()
dfs = []

for category in df['category_encoded'].unique():
    class_subset = df[df['category_encoded'] == category]
    class_upsampled = resample(class_subset, replace=True, n_samples=max_count, random_state=42)
    dfs.append(class_upsampled)

df_balanced = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)

---

## 🏗️ 5. Model Architecture (Two-Stream CNN + Transformer)

```markdown
## 🏗️ Model Architecture

The model splits the input image into upper and lower halves:
- Applies separate CNN pipelines to each half
- Uses **Multi-Head Self-Attention** on the combined feature embeddings

🖼️ Here's the architecture:

![Model Architecture](model.png)

You can also find the model saved as:
- `model_architecture.json`
- `model_weights.weights.h5`
