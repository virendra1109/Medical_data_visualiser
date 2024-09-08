import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def draw_cat_plot():
    # Load the data
    df = pd.read_csv('medical_examination.csv')
    
    # Add the overweight column
    df['height_m'] = df['height'] / 100  # Convert height to meters
    df['BMI'] = df['weight'] / (df['height_m'] ** 2)
    df['overweight'] = (df['BMI'] > 25).astype(int)
    
    # Normalize cholesterol and glucose
    df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
    df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)
    
    # Convert data to long format
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    order = ['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
    #convertung a count column for plotting
    df_cat['count']=1
    # Draw the categorical plot
    g = sns.catplot(x='variable', hue='value', col='cardio', data=df_cat, kind='count', height=4, aspect=1.5, order=order)
    
    # Adjust y-axis label to 'total'
    g.set_axis_labels("variable", "total")

    # Get the figure for the output
    fig = g.fig
    return fig
def draw_heat_map():
    # Load the data
    df = pd.read_csv('medical_examination.csv')
    
    # Add the overweight column
    df['height_m'] = df['height'] / 100  # Convert height to meters
    df['BMI'] = df['weight'] / (df['height_m'] ** 2)
    df['overweight'] = (df['BMI'] > 25).astype(int)
    
    # Normalize cholesterol and glucose
    df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
    df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)
    
    # Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]
    
    # Calculate the correlation matrix
    corr = df_heat[['id', 'age', 'sex', 'height', 'weight', 'ap_hi', 'ap_lo', 
                    'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio', 'overweight']].corr()
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))
    
    # Draw the heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap='coolwarm',center=0)
    
    # Get the figure for the output
    fig = plt.gcf()
    return fig