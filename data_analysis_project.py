import pandas as pd
import matplotlib.pyplot as plt

# === Load and Clean Data ===

# Load datasets
life_expectancy_data = pd.read_csv('Life Expectancy Data.csv')
happiness_2019 = pd.read_csv('2019.csv')
suicide_data = pd.read_csv('/Users/zachpearson/Downloads/master.csv')
# Strip whitespace from column names
life_expectancy_data.columns = life_expectancy_data.columns.str.strip()
happiness_2019.columns = happiness_2019.columns.str.strip()
suicide_data.columns = suicide_data.columns.str.strip()

# Rename columns for consistency
life_expectancy_data.rename(columns={'Year': 'year'}, inplace=True)
happiness_2019.rename(columns={'Country or region': 'Country', 'Score': 'Happiness Score'}, inplace=True)
suicide_data.rename(columns={'country': 'Country'}, inplace=True)

# Check for missing columns in Life Expectancy Data
required_columns = ['Country', 'year', 'Life expectancy', 'Alcohol']
missing_columns = [col for col in required_columns if col not in life_expectancy_data.columns]
if missing_columns:
    raise KeyError(f"Missing required columns in Life Expectancy Data: {missing_columns}")

# Clean Life Expectancy Data
life_expectancy_data = life_expectancy_data.drop_duplicates()
life_expectancy_data.ffill(inplace=True)

# Clean Suicide Data
suicide_data = suicide_data.drop_duplicates()
suicide_data['year'] = pd.to_numeric(suicide_data['year'], errors='coerce')
suicide_data.ffill(inplace=True)

# Clean Happiness Data
happiness_2019 = happiness_2019.drop_duplicates()
happiness_2019['Country'] = happiness_2019['Country'].str.strip()

# Filter datasets for the selected countries
selected_countries = ['Portugal', 'Paraguay', 'Mauritius', 'Turkmenistan', 'France',
                      'Slovakia', 'Norway', 'Sri Lanka', 'Montenegro', 'Chile']

life_expectancy_filtered = life_expectancy_data[life_expectancy_data['Country'].isin(selected_countries)]
suicide_filtered = suicide_data[suicide_data['Country'].isin(selected_countries)]
happiness_filtered = happiness_2019[happiness_2019['Country'].isin(selected_countries)]

# Convert 'year' to integer in filtered DataFrame using .loc
life_expectancy_filtered.loc[:, 'year'] = life_expectancy_filtered['year'].astype(int)

# Merge Life Expectancy with Suicide Data
merged_data = pd.merge(
    life_expectancy_filtered[['Country', 'year', 'Life expectancy', 'Alcohol']],
    suicide_filtered[['Country', 'year', 'suicides/100k pop']],
    on=['Country', 'year'],
    how='inner'
)

# Rename suicide column and Life Expectancy column
merged_data.rename(columns={'suicides/100k pop': 'Suicide Rate', 'Life expectancy': 'Life Expectancy'}, inplace=True)

# Merge with Happiness Data
merged_data = pd.merge(
    merged_data,
    happiness_filtered[['Country', 'Happiness Score']],
    on='Country',
    how='inner'
)

# Drop unused columns for the graphs
merged_data = merged_data[['Country', 'year', 'Life Expectancy', 'Alcohol', 'Suicide Rate', 'Happiness Score']]

# === Pivoting and Stacking ===

# Pivot: Average metrics by country and year
pivot_table = merged_data.pivot_table(
    values=['Life Expectancy', 'Alcohol', 'Suicide Rate', 'Happiness Score'],
    index='Country',
    columns='year',
    aggfunc='mean'
)

# Stacking: Convert pivoted data to a stacked format
stacked_data = pivot_table.stack()

# Histogram: Distribution of Life Expectancy
plt.figure(figsize=(10, 6))
plt.hist(
    merged_data['Life Expectancy'], bins=10, color='lightgreen', edgecolor='black', alpha=0.7
)
plt.title('Distribution of Life Expectancy', fontsize=16)
plt.xlabel('Life Expectancy (years)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 2. Pie Chart: 
suicide_data_by_country = merged_data.groupby('Country')['Suicide Rate'].sum().sort_values(ascending=False)
top_countries = suicide_data_by_country.head(5)
others = suicide_data_by_country.iloc[5:].sum()
suicide_pie_data = pd.concat([top_countries, pd.Series({'Others': others})])

plt.figure(figsize=(8, 8))
plt.pie(
    suicide_pie_data,
    labels=suicide_pie_data.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=plt.cm.Paired(range(len(suicide_pie_data)))
)
plt.title('Countries with High Suicide Rates', fontsize=16)
plt.show()

# 3. Box Plot: Suicide Rates by Happiness Bins
happiness_bins = pd.cut(
    merged_data['Happiness Score'],
    bins=[0, 4, 5, 6, 7, 10],
    labels=['0-4', '4-5', '5-6', '6-7', '7-10']
)
merged_data['Happiness Bin'] = happiness_bins

plt.figure(figsize=(12, 8))
box_data = [merged_data[merged_data['Happiness Bin'] == bin]['Suicide Rate'] for bin in happiness_bins.cat.categories]
plt.boxplot(
    box_data,
    patch_artist=True,
    labels=happiness_bins.cat.categories,
    boxprops=dict(facecolor='lightblue', color='blue'),
    whiskerprops=dict(color='blue'),
    capprops=dict(color='blue'),
    medianprops=dict(color='red')
)

plt.title('Suicide Rates by Happiness Bins', fontsize=16)
plt.xlabel('Happiness Bins', fontsize=12)
plt.ylabel('Suicide Rate (per 100k population)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 4. Line Chart: Average Life Expectancy vs Suicide Rates Over Time
average_data = merged_data.groupby('year').agg({
    'Life Expectancy': 'mean',
    'Suicide Rate': 'mean'
}).reset_index()

plt.figure(figsize=(12, 8))
plt.plot(
    average_data['year'], average_data['Life Expectancy'], marker='o', linestyle='-', color='purple', linewidth=2, label='Avg Life Expectancy'
)
plt.plot(
    average_data['year'], average_data['Suicide Rate'], marker='s', linestyle='--', color='orange', linewidth=2, label='Avg Suicide Rate'
)

plt.title('Average Life Expectancy and Suicide Rates Over Time', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Average Value', fontsize=12)
plt.legend(title="Metric", fontsize=10)
plt.grid()
plt.tight_layout()
plt.show()

# 5. Scatter Plot: Happiness Score vs Alcohol Consumption
scatter_data = merged_data.groupby('Country').agg({
    'Happiness Score': 'mean',
    'Alcohol': 'mean'
}).reset_index()

countries = scatter_data['Country']
colors = plt.cm.tab20(range(len(countries)))

plt.figure(figsize=(12, 8))
for i, country in enumerate(countries):
    data = scatter_data[scatter_data['Country'] == country]
    plt.scatter(data['Happiness Score'], data['Alcohol'], label=country, color=colors[i], s=100)

plt.title('Happiness Score vs Alcohol Consumption by Country', fontsize=16)
plt.xlabel('Happiness Score', fontsize=12)
plt.ylabel('Alcohol Consumption (per capita)', fontsize=12)
plt.legend(title="Country", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()