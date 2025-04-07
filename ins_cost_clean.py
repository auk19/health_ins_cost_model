import pandas as pd


df = pd.read_csv("insurance.csv")

df.head()
df.rename(columns={"charges": "cost"}, inplace=True)

## remove null values as number is low
df.isnull().sum()
df.dropna(inplace=True)

## check basic info for outliers
## standardize categories/data-types
df.info()
df.describe()

## make - values +
num_columns = df.select_dtypes(include=["number"]).columns
df[num_columns] = df[num_columns].abs()

### standardize 'sex' column
df["sex"].unique()
FEMALE = "F"
MALE = "M"
std_sex = {
    "female": FEMALE,
    "woman": FEMALE,
    "F": FEMALE,
    "male": MALE,
    "man": MALE,
    "M": MALE,
}
df["sex"] = df["sex"].replace(std_sex)

### standardize 'smoker' column
df["smoker"].unique()
# df["smoker"] = df["smoker"] == "yes"

### standardize 'region' column
df["region"].unique()
df["region"] = df["region"].str.lower()

### standardize 'cost' column
df["cost"].sample(10)
df["cost"] = df["cost"].str.strip("$").astype("float64")

df.info()
filt = df.isnull().any(axis=1)
df[filt]
df.dropna(inplace=True)

df.to_csv("cleaned-ins.csv", index=False)
