# Semantic Search in Articles using NLP

## Table of Contents

1. [Introduction](#introduction)
2. [Data Description](#data-description)
3. [Baseline Experiments](#baseline-experiments)
4. [Other Experiments](#other-experiments)
5. [Overall Conclusion](#overall-conclusion)
6. [Tools and Libraries Used](#tools-and-libraries-used)
7. [External Resources](#external-resources)
8. [Reflection Questions](#reflection-questions)
9. [Setup and Usage](#setup-and-usage)
10. [Contributing](#contributing)
11. [Author](#author)


## Introduction

This project aims to develop a semantic search pipeline for extracting relevant keywords from English articles and evaluating different models for text similarity. The primary focus is to compare traditional NLP methods with deep learning models to determine which provides the best performance in retrieving relevant information.

## Data Description

The dataset used is from Kaggle's [Movies Similarity](https://www.kaggle.com/datasets/devendra45/movies-similarity). It includes:
- **rank**: Rank of the movie based on ratings or popularity.
- **title**: Title of the movie.
- **genre**: Genre(s) of the movie (e.g., Action, Comedy, Drama).
- **wiki_plot**: Plot summary from Wikipedia.
- **imdb_plot**: Plot summary from IMDb.

We focused on `wiki_plot` and `imdb_plot` for textual data, and split the data into training (70%), validation (15%), and testing (15%) sets.

## Baseline Experiments

### Goal
To establish a baseline for semantic search using traditional text processing methods.

### Experiments
1. **Data Preprocessing**: Tokenization and cleaning using spaCy.
2. **Feature Extraction**: Building a Bag-of-Words (BoW) model, creating a dictionary, and filtering terms.
3. **Model**: TF-IDF and LSI models for text similarity.

### Conclusions
The baseline models provided initial results and insights into the effectiveness of traditional text processing methods for semantic search.

## Other Experiments

### Experiment 1: Optimizing Search with LSI Model
**Goal**: Enhance search results using the LSI model for capturing latent semantic structures.

**Steps**:
1. Built the TF-IDF model.
2. Trained the LSI model.
3. Created a similarity index for efficient query processing.

**Results**: Improved retrieval of relevant documents compared to baseline methods.

### Experiment 2: Deep Learning Approach
**Goal**: Compare LSI model results with a deep learning-based semantic search approach.

**Steps**:
1. Used Sentence-BERT (LaBSE) to encode the corpus and queries.
2. Implemented semantic search using cosine similarity on encoded embeddings.
3. Evaluated and compared with the LSI model.

**Results**: LaBSE demonstrated superior performance in understanding semantic meaning and retrieving relevant documents.

## Overall Conclusion

The project demonstrated effective semantic search using both traditional and deep learning models. The LaBSE model outperformed the LSI model, providing more accurate and relevant search results.

## Tools and Libraries Used

- **NumPy**: Numerical operations
- **pandas**: Data manipulation
- **spaCy**: Tokenization and preprocessing
- **Gensim**: Topic modeling and similarity computations
- **Matplotlib**: Visualization
- **WordCloud**: Word cloud generation
- **Sentence-Transformers**: Deep learning-based semantic search

## External Resources

- Kaggle: [Movies Similarity](https://www.kaggle.com/datasets/devendra45/movies-similarity)
- [Semantic Search using NLP](https://medium.com/analytics-vidhya/semantic-search-engine-using-nlp-cec19e8cfa7e)

## Reflection Questions

### What was the biggest challenge you faced during this project?
Ensuring preprocessing steps were effective in removing noise while retaining meaningful information was the biggest challenge.

### What do you think you have learned from the project?
I learned to preprocess textual data, build feature extraction models, and implement semantic search systems. Additionally, I gained experience in optimizing search time and evaluating model performance.

## Setup and Usage

1. **Clone the Repository**
   ```bash
   git clone https://github.com/elsayedelmandoh/LSI-LaBSE-semantic-search-widebot.git
   cd LSI-LaBSE-semantic-search-widebot
   ```
   
## Contributing

Contributions are welcome! If you have suggestions, improvements, or additional content to contribute, feel free to open issues, submit pull requests, or provide feedback. 

[![GitHub watchers](https://img.shields.io/github/watchers/elsayedelmandoh/naive-bayes-LSTM-for-sentiment-analysis-NLP-widebot.svg?style=social&label=Watch)](https://GitHub.com/elsayedelmandoh/naive-bayes-LSTM-for-sentiment-analysis-NLP-widebot/watchers/?WT.mc_id=academic-105485-koreyst)
[![GitHub forks](https://img.shields.io/github/forks/elsayedelmandoh/naive-bayes-LSTM-for-sentiment-analysis-NLP-widebot.svg?style=social&label=Fork)](https://GitHub.com/elsayedelmandoh/naive-bayes-LSTM-for-sentiment-analysis-NLP-widebot/network/?WT.mc_id=academic-105485-koreyst)
[![GitHub stars](https://img.shields.io/github/stars/elsayedelmandoh/naive-bayes-LSTM-for-sentiment-analysis-NLP-widebot.svg?style=social&label=Star)](https://GitHub.com/elsayedelmandoh/naive-bayes-LSTM-for-sentiment-analysis-NLP-widebot/stargazers/?WT.mc_id=academic-105485-koreyst)

## Author

This repository is maintained by Elsayed Elmandoh, an AI Engineer. You can connect with Elsayed on [LinkedIn and Twitter/X](https://linktr.ee/elsayedelmandoh) for updates and discussions related to Machine learning, deep learning and NLP.

Happy coding!

