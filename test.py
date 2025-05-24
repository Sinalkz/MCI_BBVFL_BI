import syft as sy
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 

data_site = sy.orchestra.launch(name="cancer-research-centre", reset=True)
client = data_site.login(email="info@openmined.org", password="changethis")
  
# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets

# metadata 
metadata = breast_cancer_wisconsin_diagnostic.metadata
# variable information 
variables = breast_cancer_wisconsin_diagnostic.variables

print(X.head(n=5))
print(X.shape)
print(y.sample(n=5, random_state=10))

# fix seed for reproducibility
SEED = 12345
np.random.seed(SEED)

X_mock = X.apply(lambda s: s + np.mean(s) + np.random.uniform(size=len(s)))
y_mock = y.sample(frac=1, random_state=SEED).reset_index(drop=True)

features_asset = sy.Asset(
    name="Breast Cancer Data: Features",
    data = X,      # real data
    mock = X_mock  # mock data
)

targets_asset = sy.Asset(
    name="Breast Cancer Data: Targets",
    data = y,      # real data
    mock = y_mock  # mock data
)

print(features_asset.data.head(n=3))
print(features_asset.mock.head(n=3))

# Metadata
description = f'{metadata["abstract"]}\n{metadata["additional_info"]["summary"]}'

paper = metadata["intro_paper"]
citation = f'{paper["authors"]} - {paper["title"]}, {paper["year"]}'

summary = "The Breast Cancer Wisconsin dataset can be used to predict whether the cancer is benign or malignant."

# Dataset creation
breast_cancer_dataset = sy.Dataset(
    name="Breast Cancer Biomarker",
    description=description,
    summary=summary,
    citation=citation,
    url=metadata["dataset_doi"],
)

breast_cancer_dataset.add_asset(features_asset)

breast_cancer_dataset.add_asset(targets_asset)

print(breast_cancer_dataset)
print(client.datasets)

client.upload_dataset(dataset=breast_cancer_dataset)

data_site.land()