# 🎬 Sentiment Analysis of Movie Reviews Using Machine Learning

Classifies IMDb movie reviews as **Positive** or **Negative** using TF-IDF features and three classic ML models
(Multinomial Naive Bayes, Logistic Regression, Linear SVM), tuned with GridSearchCV.

**Dataset:** [IMDb Dataset of 50K Movie Reviews (Kaggle)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
(50,000 reviews, balanced; 418 duplicates removed -> 49,582 used)

## 📁 Project Structure

```text
ML Project/
├── main.py                      # Runs the full pipeline (or a single step)
├── app.py                       # Web app (Flask) to try the model in your browser
├── templates/
│   └── index.html               # Web page used by app.py
├── src/
│   ├── config.py                # Paths, constants, TF-IDF settings
│   ├── data_preprocessing.py    # Load, clean, encode labels, train/test split
│   ├── eda.py                   # EDA plots + Mann-Whitney / Chi-square tests
│   ├── train.py                 # Baseline + GridSearchCV training, saves models
│   ├── evaluate.py              # Metrics, confusion matrices, ROC, feature importance
│   └── predict.py               # Predict sentiment for new reviews
├── Sentiment_Analysis_of_Movie_Reviews_Using_Machine_Learning.ipynb   # Original notebook
├── IMDB Dataset.csv             # Download from Kaggle (not included in this repo copy)
├── requirements.txt
└── outputs/                     # Created when you run the code
    ├── best_sentiment_model.pkl
    ├── all_models.pkl
    ├── cv_results.csv
    ├── model_comparison.csv
    ├── test_predictions.csv
    └── figures/                 # All charts (.png)
```

## 🚀 How to Run

```bash
pip install -r requirements.txt

python main.py                   # preprocess -> EDA -> train -> evaluate
python main.py --step eda        # only EDA
python main.py --step train      # only training (saves models)
python main.py --step evaluate   # only evaluation (needs trained models)
python main.py --sample 5000     # quick test run on 5,000 reviews
```

Predict on new text after training:

```bash
python -m src.predict "This movie was fantastic!"
python -m src.predict            # interactive mode
```

### Web app

After training (`python main.py --step train`), start the app:

```bash
python app.py
```

Open http://127.0.0.1:5000, paste a movie review and click **Analyze sentiment**.
The app also accepts JSON: `POST /predict` with `{"review": "text"}`.

Run commands from the project root folder (the one containing `main.py`).

## ⚙️ Workflow

1. **Preprocessing** (`data_preprocessing.py`) – remove duplicates, strip `<br />` tags, map labels to 0/1.
2. **EDA** (`eda.py`) – class balance, review length, top words, correlation, Mann-Whitney U and Chi-square tests.
3. **Training** (`train.py`) – majority-class baseline, then TF-IDF + each model tuned with 3-fold GridSearchCV (F1).
   The best model by **cross-validation F1** is saved (not chosen on the test set).
4. **Evaluation** (`evaluate.py`) – accuracy, precision, recall, F1, ROC-AUC, confusion matrices, ROC curves,
   top positive/negative words from Logistic Regression.
5. **Prediction** (`predict.py`) – load the saved model and classify new reviews.

## 📊 Results (from the original notebook run)

| Model | CV F1 | Test Accuracy | Test F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Naive Bayes | 0.8763 | 0.8799 | 0.8814 | 0.9480 |
| Logistic Regression | 0.8989 | 0.9027 | 0.9039 | 0.9667 |
| **Linear SVM** | **0.9004** | **0.9039** | **0.9052** | 0.9664 |

Majority-class baseline accuracy: 50.19%.

## ⚠️ Limitations

- Trained only on IMDb reviews; may not generalize to other kinds of text.
- Sarcasm and complex expressions are hard for TF-IDF models.
- Review length differs only slightly between classes (a statistically significant but tiny effect).

## 🔮 Future Improvements

- LSTM / Bi-LSTM and BERT-based models.
- Deploy the web app online (e.g. Render or Hugging Face Spaces).

## 👨‍💻 Author

**Pawan Dubey** – [@pawandubey82](https://github.com/pawandubey82)
