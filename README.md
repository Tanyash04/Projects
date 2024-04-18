# Baseball Performance Prediction

This project aims to predict baseball player performance based on various hitting statistics. The code includes data loading, preprocessing, feature engineering, and model training for predictive analytics.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Data](#data)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In baseball, evaluating player performance is crucial for team management and player development. This project focuses on predicting a player's performance (measured by runs scored) using hitting statistics such as at-bats (AB), hits (H), doubles (2B), triples (3B), home runs (HR), walks (BB), strikeouts (SO), stolen bases (SB), and caught stealing (CS). Additionally, three common metrics in baseball, namely Batting Average (BA), On-Base Percentage (OBP), and Slugging Percentage (SLG), are calculated and used as features for prediction.

## Features

The following hitting statistics are used as features for prediction:

- AB: At-bats
- H: Hits
- 2B: Doubles
- 3B: Triples
- HR: Home runs
- BB: Walks
- SO: Strikeouts
- SB: Stolen bases
- CS: Caught stealing
- BA: Batting Average
- OBP: On-Base Percentage
- SLG: Slugging Percentage

## Data

The data used in this project is stored in the following CSV files:
- `train_C.csv`: Training data containing hitting statistics for baseball players.
- `test_set_labeled.csv`: Test data for evaluating the trained models.

## Usage

To use this project, follow these steps:
1. Clone this repository to your local machine.
2. Install the required dependencies listed in `requirements.txt`.
3. Run the Python scripts provided in the repository.
4. Follow the instructions provided in the scripts for data loading, preprocessing, model training, and evaluation.

## Models

Several machine learning models are employed for performance prediction:
- Random Forest Regressor
- Neural Network (Multi-layer Perceptron)
- Support Vector Regression (SVR)
- K-Nearest Neighbors (KNN)
- Gradient Boosting Regressor
- Gaussian Process Regression
- Decision Tree Regressor
- Non-linear Regression

## Results

The performance of each model is evaluated using Mean Squared Error (MSE) on the test data. Results and insights from model evaluation are provided in the code and documentation.

## Contributing

Contributions to this project are welcome! If you have any ideas for improvements, feel free to open an issue or submit a pull request.



