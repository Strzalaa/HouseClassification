# === SECTION: Imports ===
import os
import sys
import requests
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from category_encoders import TargetEncoder

# === SECTION: Data Folder & Download Setup ===
if __name__ == '__main__':
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    if 'house_class.csv' not in os.listdir('../Data'):
        sys.stderr.write("[INFO] Dataset is loading.\n")
        dataset_url = "https://www.dropbox.com/s/7vjkrlggmvr5bc1/house_class.csv?dl=1"
        dataset_response = requests.get(dataset_url, allow_redirects=True)
        open('../Data/house_class.csv', 'wb').write(dataset_response.content)
        sys.stderr.write("[INFO] Loaded.\n")

# === SECTION: Load Dataset ===
raw_housing_df = pd.read_csv("/Users/strzala/Downloads/house_class.csv")
feature_data_df = raw_housing_df[['Area', 'Room', 'Lon', 'Lat', 'Zip_area', 'Zip_loc']]
target_price_series = raw_housing_df["Price"]

# === SECTION: Train/Test Split ===
X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(
    feature_data_df,
    target_price_series,
    test_size=0.3,
    random_state=1,
    stratify=feature_data_df['Zip_loc']
)

categorical_columns = ['Zip_area', 'Zip_loc', 'Room']

X_train_ohe = X_train_data.copy()
X_test_ohe = X_test_data.copy()
X_train_ord = X_train_data.copy()
X_test_ord = X_test_data.copy()
X_train_target = X_train_data.copy()
X_test_target = X_test_data.copy()

# === SECTION: OneHot Encoding ===
encoder_ohe = OneHotEncoder(drop='first')
encoder_ohe.fit(X_train_ohe[categorical_columns])

X_train_ohe_encoded = pd.DataFrame(
    encoder_ohe.transform(X_train_ohe[categorical_columns]).toarray(),
    index=X_train_ohe.index
)
X_test_ohe_encoded = pd.DataFrame(
    encoder_ohe.transform(X_test_ohe[categorical_columns]).toarray(),
    index=X_test_ohe.index
)

X_train_ohe_final = X_train_ohe[['Area', 'Lon', 'Lat']].join(X_train_ohe_encoded)
X_test_ohe_final = X_test_ohe[['Area', 'Lon', 'Lat']].join(X_test_ohe_encoded)

X_train_ohe_final.columns = X_train_ohe_final.columns.astype(str)
X_test_ohe_final.columns = X_test_ohe_final.columns.astype(str)

# === SECTION: Train Model with OneHotEncoder ===
model_ohe = DecisionTreeClassifier(
    criterion='entropy',
    max_features=3,
    splitter='best',
    max_depth=6,
    min_samples_split=4,
    random_state=3
)
model_ohe.fit(X_train_ohe_final, y_train_data)
y_pred_ohe = model_ohe.predict(X_test_ohe_final)
acc_ohe = accuracy_score(y_test_data, y_pred_ohe)

# === SECTION: Ordinal Encoding ===
encoder_ord = OrdinalEncoder()
encoder_ord.fit(X_train_ord[categorical_columns])

X_train_ord[categorical_columns] = encoder_ord.transform(X_train_ord[categorical_columns])
X_test_ord[categorical_columns] = encoder_ord.transform(X_test_ord[categorical_columns])

X_train_ord_final = X_train_ord[['Area', 'Lon', 'Lat']].join(X_train_ord[categorical_columns])
X_test_ord_final = X_test_ord[['Area', 'Lon', 'Lat']].join(X_test_ord[categorical_columns])

X_train_ord_final.columns = X_train_ord_final.columns.astype(str)
X_test_ord_final.columns = X_test_ord_final.columns.astype(str)

# === SECTION: Train Model with OrdinalEncoder ===
model_ord = DecisionTreeClassifier(
    criterion='entropy',
    max_features=3,
    splitter='best',
    max_depth=6,
    min_samples_split=4,
    random_state=3
)
model_ord.fit(X_train_ord_final, y_train_data)
y_pred_ord = model_ord.predict(X_test_ord_final)
acc_ord = accuracy_score(y_test_data, y_pred_ord)

# === SECTION: Target Encoding ===
pd.set_option('future.no_silent_downcasting', True)

tencoder = TargetEncoder(cols=['Room', 'Zip_area', 'Zip_loc'])
tencoder.fit(X_train_target[['Room', 'Zip_area', 'Zip_loc']], y_train_data)

X_train_target_transformed = tencoder.transform(X_train_target[['Room', 'Zip_area', 'Zip_loc']])
X_test_target_transformed = tencoder.transform(X_test_target[['Room', 'Zip_area', 'Zip_loc']])

X_train_target_final = X_train_target[['Area', 'Lon', 'Lat']].join(X_train_target_transformed)
X_test_target_final = X_test_target[['Area', 'Lon', 'Lat']].join(X_test_target_transformed)

dtarget = DecisionTreeClassifier(
    criterion='entropy',
    max_features=3,
    splitter='best',
    max_depth=6,
    min_samples_split=4,
    random_state=3
)
dtarget.fit(X_train_target_final, y_train_data)
y_pred_target = dtarget.predict(X_test_target_final)
accuracytarget = accuracy_score(y_test_data, y_pred_target)

# === SECTION: Model Evaluation ===
report_ohe_dict = classification_report(y_test_data, y_pred_ohe, output_dict=True)
report_ord_dict = classification_report(y_test_data, y_pred_ord, output_dict=True)
report_target_dict = classification_report(y_test_data, y_pred_target, output_dict=True)

print(f"OneHotEncoder:{round(report_ohe_dict['macro avg']['f1-score'], 2)}")
print(f"OrdinalEncoder:{round(report_ord_dict['macro avg']['f1-score'], 2)}")
print(f"TargetEncoder:{round(round(report_target_dict['macro avg']['f1-score'], 4), 3)}")
