# Road Blackspot Prediction

Predicting which road locations become crash **blackspots** from the socio-economic and land-use profile of the surrounding area, using an interpretable logistic regression model.

## Why it matters

Road authorities have limited budgets for safety upgrades. A model that flags likely blackspots — and explains *which* local factors drive the risk — helps prioritise where to invest before crashes happen.

## Approach

1. **Clean** — impute missing values (median for age share, mean for liquor licences) and encode `Intersection` and `Blackspot` as 0/1.
2. **Explore** — check the class balance of blackspot vs. non-blackspot locations.
3. **Select features** — area disadvantage (`SEIFA`), share of households without a car, liquor licence density, and supermarket presence.
4. **Model** — stratified 80/20 split, standard scaling fitted on training data only, logistic regression.
5. **Evaluate** — accuracy, precision, recall, F1, ROC AUC and the ROC curve, plus model coefficients to show each feature's effect.

## Features used

| Feature | Meaning |
| --- | --- |
| `SEIFA` | Socio-economic index for the area |
| `CARS_ZERO_HH_PCNT` | % of households with no car |
| `Lq_Licenses` | Liquor licences nearby |
| `Supermarket` | Supermarket presence nearby |

## Getting started

```bash
git clone https://github.com/kr-xna/road-blackspot-prediction.git
cd road-blackspot-prediction
pip install -r requirements.txt
python blackspot_prediction.py
```

Place `Blackspot.csv` in the project folder first. The dataset is not included in this repository.

## Tech stack

Python · pandas · scikit-learn · Matplotlib · seaborn

---

Built for the *Machine Learning in Business* unit of the Master of Business Analytics at Deakin University.
