import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import the data from 'medical_examination.csv'
df = pd.read_csv('medical_examination.csv')

# 2. Add an overweight column (BMI > 25 is overweight)
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (df['BMI'] > 25).astype(int)  # 1 if overweight, 0 otherwise

# 3. Normalize cholesterol and gluc columns (1 -> 0, > 1 -> 1)
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4. Draw the Categorical Plot
def draw_cat_plot():
    # Melt the DataFrame for categorical plot
    df_cat = pd.melt(df, id_vars='cardio', value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # Rename columns for clarity
    df_cat.rename(columns={'variable': 'feature', 'value': 'value'}, inplace=True)
    
    # Create the categorical plot using seaborn
    fig = sns.catplot(x='feature', hue='value', col='cardio', data=df_cat, kind='count')
    
    # Adjust the plot for better readability
    fig.set_axis_labels('Feature', 'Count')
    plt.xticks(rotation=45)
    
    return fig

# 5. Draw the Heat Map
def draw_heat_map():
    # Clean the data for heatmap by filtering incorrect data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]
    
    # Calculate the correlation matrix
    corr = df_heat.corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(corr)
    
    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))
    
    # Plot the heatmap
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt='.1f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

# Do not modify the next two lines.
# These are for testing purposes and will be handled by main.py and test_module.py.
