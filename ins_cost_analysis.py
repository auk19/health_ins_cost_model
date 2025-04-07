import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("cleaned_ins.csv")

## distribution of ins cost
plt.hist(df["cost"], bins=20, color="gray", edgecolor="black")
plt.title("Distribution of Insurance Costs")
plt.xlabel("Cost")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

## age | cost
plt.scatter(df["age"], df["cost"])
plt.title("Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Cost")
plt.grid(True)
plt.show()

## bmi | cost
plt.scatter(df["bmi"], df["cost"])
plt.title("Cost vs BMI")
plt.xlabel("BMI")
plt.ylabel("Cost")
plt.grid(True)
plt.show()

## smoker | cost
smoker_avg_cost = df.groupby("smoker")["cost"].mean()
plt.bar(smoker_avg_cost.index, smoker_avg_cost.values, color=["red", "green"])
plt.title("Cost vs Smoker Status")
plt.xlabel("Smoker")
plt.ylabel("Cost")
plt.show()

## region | cost
region = df.groupby("region")["cost"].mean()
plt.bar(region.index, region.values)
plt.title("Cost vs Region")
plt.xlabel("Region")
plt.ylabel("Cost")
plt.show()
