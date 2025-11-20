# Spam Classifier using Support Vector Machine (SVM)

This repository contains a machine learning project for classifying emails or messages as spam or not spam using a Support Vector Machine (SVM) algorithm. The implementation is done in Python, making use of various popular libraries for data processing, model training, and evaluation.

## Features

- **Email/Text preprocessing:** Cleans and prepares input data for classification.
- **Feature Extraction:** Converts text into numerical features suitable for machine learning algorithms (e.g., using TF-IDF).
- **SVM Classifier:** Utilizes the Support Vector Machine algorithm for robust spam detection.
- **Model Evaluation:** Measures performance using standard metrics such as accuracy, precision, recall, and confusion matrix.
- **Example usage:** Includes sample code for training and testing the classifier.

## Getting Started

### Prerequisites

- Python 3.x
- Recommended libraries:
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `matplotlib` (for visualization, optional)

Install dependencies using pip:

```bash
pip install scikit-learn pandas numpy matplotlib
```

### Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/skloniscasan/Spam-Classifier----Support-Vector-Machine.git
   cd Spam-Classifier----Support-Vector-Machine
   ```

2. **Prepare the dataset:**

   - Place your dataset (e.g., a CSV file with labeled spam and ham messages) in the project directory.
   - Make sure you update the file path in the code if necessary.

3. **Run the classifier:**

   ```bash
   python spam_classifier.py
   ```

   (Replace `spam_classifier.py` with the actual file name if different.)

4. **Results:**
   - The script loads the data, preprocesses it, trains an SVM model, evaluates the performance, and may display results/plots.

## File Structure

- `spam_classifier.py` &mdash; Main script for loading data, training SVM, and evaluating the classifier.
- `README.md` &mdash; Project documentation.
- (Optionally) `requirements.txt` &mdash; List of dependencies.

## How it Works

1. **Data Loading:** Loads a labeled dataset containing messages and their spam/ham labels.
2. **Preprocessing:** Cleans text by removing unwanted characters, lowercasing, and possibly stemming/lemmatizing.
3. **Feature Extraction:** Transforms messages into numerical features (e.g., using TF-IDF Vectorizer).
4. **Training:** The SVM classifier is trained on the extracted features.
5. **Prediction & Evaluation:** The model predicts spam or ham labels on the test set and evaluates its performance with metrics and confusion matrix.

## Example

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Example data loading step (replace with actual file or DataFrame)
import pandas as pd
data = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
data.columns = ['label', 'message']

# Preprocess and encode labels
data['label_num'] = data.label.map({'ham':0, 'spam':1})

X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label_num'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

clf = SVC(kernel='linear')
clf.fit(X_train_vec, y_train)

y_pred = clf.predict(X_test_vec)
print(classification_report(y_test, y_pred))
```

## Contributing

Contributions are welcome! Please open issues or create pull requests if you have suggestions or improvements.

## License

This project is licensed under the MIT License.

---

**Author:** [skloniscasan](https://github.com/skloniscasan)
