# Domain-Dependent Hybrid Recommender System

Comparative study of recommendation algorithms across movies and books domains with transfer learning evaluation. Bachelor thesis project.

## What It Does

Implements and compares SVD, item-based collaborative filtering, content-based filtering, and hybrid approaches. Includes cross-domain transfer learning experiments.

Main finding: Content-based methods outperformed collaborative filtering in both domains, and books-to-movies transfer worked better than the reverse.

## Requirements

pandas, numpy, scikit-learn, scipy, matplotlib, seaborn

## Datasets
Datasets can be found here https://drive.google.com/file/d/1dQSh9anfRPYk3M7JQp-yvcpFr36l9BF0/view?usp=sharing, please put the "Datasets" folder in the root directory of the project.
- `Datasets/The Movies Dataset/` - movies_clean.csv, content_features.csv, ratings_clean.csv
- `Datasets/goodbooks_10k_rating_and_description/` - goodbooks_10k_rating_and_description.csv, ratings,csv

## Usage

Choose from menu:

Movies pipeline - model training and evaluation
Books pipeline - model training and evaluation
Cross-domain comparison - model training and evaluation of both domains, plot output
Transfer learning - implement and test knowledge transfer between domains

## Key Results

Content-based: P@10 0.21-0.33
Collaborative filtering: P@10 0.04-0.20
Hybrids showed minimal gains
Transfer retention: 30-80%

## Files

hybrid_rec.ipynb - Core models and pipelines
transfer_learning.py - Cross-domain transfer module
clean_movies_dataset_enrich_goodbooks.ipynb - Preprocessing tool for The Movies Dataset
