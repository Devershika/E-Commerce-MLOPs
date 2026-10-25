final_results_df = pd.DataFrame(all_results)

model_performance_final = final_results_df.groupby('Model')[['MAPE', 'R2']].mean().sort_values(by='MAPE')

print("\nOverall Model Performance (Lower MAPE is better, Higher R2 is better):\n")
print(model_performance_final)

plt.figure(figsize=(12, 6))
sns.barplot(x=model_performance_final.index, y='MAPE', data=model_performance_final, palette='viridis')
plt.title('Overall Average MAPE by Model')
plt.ylabel('Mean Absolute Percentage Error (MAPE)')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x=model_performance_final.index, y='R2', data=model_performance_final, palette='magma')
plt.title('Overall Average R2 by Model')
plt.ylabel('R-squared (R2)')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.show()

results_df_finetuned = pd.DataFrame(final_results_df)


# Mean metrics per model
print(
    results_df_finetuned
        .groupby('Model')[['MAPE', 'R2']]
        .mean()
        .sort_values('MAPE')
)

# Preview top rows
results_df_finetuned.head(20)


for (cat, reg), subset in results_df_finetuned.groupby(['Category', 'Region']):

    print(f"\nCategory: {cat} | Region: {reg} ")

    print(
        subset
        .groupby('Model')[['MAPE', 'R2']]
        .mean()
        .sort_values('MAPE')
    )

    display(subset.head(10))

def select_top_products(df, metric_col, top_n=5):
    """
    Select top N products per Category, Region, Store
    based on a metric (e.g., Demand or Forecast)
    """
    return (
        df
        .groupby(['Category', 'Region', 'Store ID', 'Product ID'], as_index=False)[metric_col]
        .mean()
        .sort_values(metric_col, ascending=False)
        .groupby(['Category', 'Region', 'Store ID'])
        .head(top_n)
        .merge(df, on=['Category', 'Region', 'Store ID', 'Product ID'], how='left')
    )


# First, identify the top products based on 'Demand' from the original df
top_products_ids = (
    df.groupby(['Category', 'Region', 'Store ID', 'Product ID'])['Demand']
    .mean()
    .reset_index()
    .sort_values('Demand', ascending=False)
    .groupby(['Category', 'Region', 'Store ID'])
    .head(5) # top_n is 5 as per the original intent
    [['Category', 'Region', 'Store ID', 'Product ID']] # Select only the identifier columns
)

# Then, filter the results_df_finetuned to include only these top products
filtered_results = results_df_finetuned.merge(
    top_products_ids,
    on=['Category', 'Region', 'Store ID', 'Product ID'],
    how='inner'
)

for (cat, reg), subset in filtered_results.groupby(['Category', 'Region']):

    print(f"Category: {cat} | Region: {reg}")


    print(
        subset
        .groupby('Model')[['MAPE', 'R2']]
        .mean()
        .sort_values('MAPE')
    )
