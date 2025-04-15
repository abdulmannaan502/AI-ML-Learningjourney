# Sample Code for Categorical Data Handling

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Example categorical data
data = pd.DataFrame({'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue']})

# One-hot encoding
encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(data[['Color']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Color']))

print(encoded_df)
