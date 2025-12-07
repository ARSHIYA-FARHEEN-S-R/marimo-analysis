"""
# Interactive Data Analysis Notebook
## Research Institution - Data Science Division

**Analyst:** 23f2001189@ds.study.iitm.ac.in
**Dataset:** Housing Price Analysis
**Created:** December 2023
"""

import marimo

__generated_with = "0.1.64"
app = marimo.App(width="full")


# ============================================================================
# CELL 1: Import Libraries and Setup
# ============================================================================
@app.cell
def _():
    """
    ## Import Required Libraries
    
    This cell imports all necessary Python libraries for data analysis,
    visualization, and interactive widgets.
    
    Contact: 23f2001189@ds.study.iitm.ac.in
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    
    # Import marimo UI components
    import marimo as mo
    from marimo import ui
    
    print("✅ Libraries imported successfully")
    return mo, np, pd, plt, sns, stats, ui


# ============================================================================
# CELL 2: Load and Explore Dataset
# ============================================================================
@app.cell
def _(pd):
    """
    ## Load Housing Price Dataset
    
    This cell loads the dataset and displays basic information.
    The dataset contains housing prices with various features.
    """
    # Create synthetic housing dataset
    np.random.seed(42)
    n_samples = 500
    
    # Generate features
    square_footage = np.random.normal(2000, 500, n_samples).clip(800, 4000)
    bedrooms = np.random.choice([2, 3, 4, 5], n_samples, p=[0.1, 0.4, 0.4, 0.1])
    bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3], n_samples)
    age = np.random.exponential(20, n_samples).clip(0, 100)
    location_score = np.random.uniform(0, 10, n_samples)
    
    # Generate target variable (price) with relationships
    base_price = 100000
    price = (
        base_price
        + 150 * square_footage
        + 50000 * bedrooms
        + 30000 * bathrooms
        - 2000 * age
        + 25000 * location_score
        + np.random.normal(0, 50000, n_samples)
    ).clip(150000, 1500000)
    
    # Create DataFrame
    data = pd.DataFrame({
        'price': price,
        'square_footage': square_footage,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age': age,
        'location_score': location_score
    })
    
    # Display dataset info
    print(f"📊 Dataset Shape: {data.shape}")
    print(f"📈 Features: {list(data.columns)}")
    
    return data, n_samples


# ============================================================================
# CELL 3: Dataset Statistics (Depends on Cell 2)
# ============================================================================
@app.cell
def _(data, mo):
    """
    ## Dataset Statistics
    
    This cell calculates basic statistics of the dataset.
    It depends on the 'data' variable from Cell 2.
    """
    stats_summary = data.describe().round(2)
    
    return mo.md(f"""
    ### 📋 Dataset Summary Statistics
    
    | Metric | Price | Sq Ft | Bedrooms | Bathrooms | Age | Location |
    |--------|-------|-------|----------|-----------|-----|----------|
    | Mean | ${stats_summary.loc['mean', 'price']:,.0f} | {stats_summary.loc['mean', 'square_footage']:.0f} | {stats_summary.loc['mean', 'bedrooms']:.1f} | {stats_summary.loc['mean', 'bathrooms']:.1f} | {stats_summary.loc['mean', 'age']:.1f} | {stats_summary.loc['mean', 'location_score']:.1f} |
    | Std | ${stats_summary.loc['std', 'price']:,.0f} | {stats_summary.loc['std', 'square_footage']:.0f} | {stats_summary.loc['std', 'bedrooms']:.1f} | {stats_summary.loc['std', 'bathrooms']:.1f} | {stats_summary.loc['std', 'age']:.1f} | {stats_summary.loc['std', 'location_score']:.1f} |
    | Min | ${stats_summary.loc['min', 'price']:,.0f} | {stats_summary.loc['min', 'square_footage']:.0f} | {stats_summary.loc['min', 'bedrooms']:.0f} | {stats_summary.loc['min', 'bathrooms']:.1f} | {stats_summary.loc['min', 'age']:.1f} | {stats_summary.loc['min', 'location_score']:.1f} |
    | Max | ${stats_summary.loc['max', 'price']:,.0f} | {stats_summary.loc['max', 'square_footage']:.0f} | {stats_summary.loc['max', 'bedrooms']:.0f} | {stats_summary.loc['max', 'bathrooms']:.1f} | {stats_summary.loc['max', 'age']:.1f} | {stats_summary.loc['max', 'location_score']:.1f} |
    
    **Data Analyst:** 23f2001189@ds.study.iitm.ac.in
    """), stats_summary


# ============================================================================
# CELL 4: Interactive Slider Widget
# ============================================================================
@app.cell
def _(ui):
    """
    ## Interactive Price Filter
    
    This cell creates a slider widget to filter properties by price range.
    The widget state will be used in subsequent cells for filtering.
    """
    price_slider = ui.slider(
        start=150000,
        stop=1500000,
        step=50000,
        value=[300000, 800000],
        label="💰 Price Range Filter"
    )
    
    bedroom_slider = ui.slider(
        start=2,
        stop=5,
        step=1,
        value=3,
        label="🛏️ Number of Bedrooms"
    )
    
    return price_slider, bedroom_slider


# ============================================================================
# CELL 5: Filter Data Based on Widgets (Depends on Cells 2 and 4)
# ============================================================================
@app.cell
def _(data, price_slider, bedroom_slider, mo):
    """
    ## Apply Filters and Display Results
    
    This cell filters the dataset based on the slider values from Cell 4.
    It demonstrates variable dependencies between cells.
    """
    # Get current slider values
    min_price, max_price = price_slider.value
    selected_bedrooms = bedroom_slider.value
    
    # Apply filters
    filtered_data = data[
        (data['price'] >= min_price) &
        (data['price'] <= max_price) &
        (data['bedrooms'] == selected_bedrooms)
    ]
    
    # Calculate statistics for filtered data
    avg_price = filtered_data['price'].mean()
    avg_sqft = filtered_data['square_footage'].mean()
    avg_age = filtered_data['age'].mean()
    count = len(filtered_data)
    
    # Dynamic markdown output based on widget state
    return mo.md(f"""
    ### 🔍 Filtered Results
    
    **Applied Filters:**
    - Price Range: ${min_price:,} - ${max_price:,}
    - Bedrooms: {selected_bedrooms}
    
    **Results:**
    - 📊 Properties Found: **{count}**
    - 💰 Average Price: **${avg_price:,.0f}**
    - 📐 Average Square Footage: **{avg_sqft:.0f} sq ft**
    - 🏠 Average Age: **{avg_age:.1f} years**
    
    **Price per Sq Ft:** ${(avg_price/avg_sqft):.0f}
    
    *Analysis by: 23f2001189@ds.study.iitm.ac.in*
    """), filtered_data, min_price, max_price, selected_bedrooms, avg_price, avg_sqft, avg_age, count


# ============================================================================
# CELL 6: Correlation Analysis (Depends on Cell 5)
# ============================================================================
@app.cell
def _(filtered_data, mo, np):
    """
    ## Correlation Analysis
    
    This cell calculates correlations between variables in the filtered dataset.
    It depends on 'filtered_data' from Cell 5.
    """
    # Calculate correlation matrix
    corr_matrix = filtered_data.corr()
    
    # Calculate individual correlations with price
    price_correlations = {}
    for col in filtered_data.columns:
        if col != 'price':
            corr = np.corrcoef(filtered_data['price'], filtered_data[col])[0, 1]
            price_correlations[col] = corr
    
    # Find strongest correlation
    strongest_corr_feature = max(price_correlations, key=price_correlations.get)
    strongest_corr_value = price_correlations[strongest_corr_feature]
    
    return mo.md(f"""
    ### 📊 Correlation Analysis
    
    **Correlation with Price:**
    
    | Feature | Correlation Coefficient |
    |---------|-------------------------|
    | Square Footage | {price_correlations.get('square_footage', 0):.3f} |
    | Bedrooms | {price_correlations.get('bedrooms', 0):.3f} |
    | Bathrooms | {price_correlations.get('bathrooms', 0):.3f} |
    | Age | {price_correlations.get('age', 0):.3f} |
    | Location Score | {price_correlations.get('location_score', 0):.3f} |
    
    **Strongest Relationship:** {strongest_corr_feature.replace('_', ' ').title()} 
    (r = {strongest_corr_value:.3f})
    
    **Interpretation:**
    - Values close to 1: Strong positive relationship
    - Values close to -1: Strong negative relationship  
    - Values near 0: Weak or no relationship
    
    *Contact: 23f2001189@ds.study.iitm.ac.in*
    """), corr_matrix, price_correlations, strongest_corr_feature, strongest_corr_value


# ============================================================================
# CELL 7: Visualizations (Depends on Cells 2, 5, 6)
# ============================================================================
@app.cell
def _(filtered_data, plt, sns, price_correlations, strongest_corr_feature, mo):
    """
    ## Data Visualization
    
    This cell creates interactive visualizations based on filtered data.
    It depends on multiple variables from previous cells.
    """
    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 10))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Price distribution
    axes[0, 0].hist(filtered_data['price'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Price ($)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Price Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Price vs strongest correlated feature
    axes[0, 1].scatter(filtered_data[strongest_corr_feature], 
                      filtered_data['price'], 
                      alpha=0.6, 
                      edgecolors='w', 
                      s=50)
    axes[0, 1].set_xlabel(strongest_corr_feature.replace('_', ' ').title())
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].set_title(f'Price vs {strongest_corr_feature.replace("_", " ").title()}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add regression line
    z = np.polyfit(filtered_data[strongest_corr_feature], filtered_data['price'], 1)
    p = np.poly1d(z)
    axes[0, 1].plot(filtered_data[strongest_corr_feature], 
                   p(filtered_data[strongest_corr_feature]), 
                   "r--", 
                   alpha=0.8,
                   label=f'r = {price_correlations[strongest_corr_feature]:.3f}')
    axes[0, 1].legend()
    
    # Plot 3: Correlation heatmap
    corr_matrix = filtered_data.corr()
    im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
    axes[1, 0].set_title('Correlation Matrix')
    axes[1, 0].set_xticks(range(len(corr_matrix.columns)))
    axes[1, 0].set_yticks(range(len(corr_matrix.columns)))
    axes[1, 0].set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    axes[1, 0].set_yticklabels(corr_matrix.columns)
    
    # Add correlation values
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text = axes[1, 0].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                  ha="center", va="center", color="w", fontsize=9)
    
    # Plot 4: Feature importance (absolute correlation)
    features = list(price_correlations.keys())
    corr_values = [abs(price_correlations[f]) for f in features]
    
    y_pos = np.arange(len(features))
    axes[1, 1].barh(y_pos, corr_values, alpha=0.7, color='steelblue')
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels([f.replace('_', ' ').title() for f in features])
    axes[1, 1].set_xlabel('Absolute Correlation with Price')
    axes[1, 1].set_title('Feature Importance')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    return mo.md(f"""
    ### 📈 Interactive Visualizations
    
    The plots below update automatically based on your filter selections:
    
    1. **Price Distribution** - Histogram of filtered property prices
    2. **Price vs {strongest_corr_feature.replace('_', ' ').title()}** - Scatter plot with regression line
    3. **Correlation Matrix** - Heatmap of relationships between all features
    4. **Feature Importance** - Bar chart of correlation magnitudes
    
    *Visualizations generated by: 23f2001189@ds.study.iitm.ac.in*
    """), fig


# ============================================================================
# CELL 8: Statistical Model (Depends on Cells 2, 5)
# ============================================================================
@app.cell
def _(filtered_data, np, stats, mo):
    """
    ## Statistical Modeling
    
    This cell performs linear regression to model price prediction.
    It depends on the filtered dataset from Cell 5.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    
    # Prepare features and target
    X = filtered_data[['square_footage', 'bedrooms', 'bathrooms', 'age', 'location_score']]
    y = filtered_data['price']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Make predictions
    y_pred = model.predict(X_scaled)
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    # Calculate coefficients (unstandardized)
    feature_names = X.columns
    coefficients = model.coef_
    intercept = model.intercept_
    
    # Create coefficient table
    coeff_table = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Absolute_Impact': np.abs(coefficients)
    }).sort_values('Absolute_Impact', ascending=False)
    
    # Calculate price prediction for average property
    avg_property = X.mean().values.reshape(1, -1)
    avg_property_scaled = scaler.transform(avg_property)
    predicted_avg_price = model.predict(avg_property_scaled)[0]
    
    return mo.md(f"""
    ### 🧮 Linear Regression Model
    
    **Model Performance:**
    - R² Score: **{r2:.4f}**
    - RMSE: **${rmse:,.0f}**
    - Intercept: **${intercept:,.0f}**
    
    **Feature Coefficients:**
    
    | Feature | Coefficient | Impact Direction |
    |---------|-------------|------------------|
    {''.join([f"| {row['Feature']} | {row['Coefficient']:,.0f} | {'Positive' if row['Coefficient'] > 0 else 'Negative'} |\n" for _, row in coeff_table.iterrows()])}
    
    **Interpretation:**
    - Each unit increase in square footage adds **${coeff_table[coeff_table['Feature'] == 'square_footage']['Coefficient'].values[0]:,.0f}** to price
    - Each additional bedroom adds **${coeff_table[coeff_table['Feature'] == 'bedrooms']['Coefficient'].values[0]:,.0f}** to price
    - Each year of age reduces price by **${abs(coeff_table[coeff_table['Feature'] == 'age']['Coefficient'].values[0]):,.0f}**
    
    **Predicted Price for Average Property:** ${predicted_avg_price:,.0f}
    
    *Model developed by: 23f2001189@ds.study.iitm.ac.in*
    """), model, r2, rmse, coefficients, intercept, coeff_table, predicted_avg_price


# ============================================================================
# CELL 9: Interactive Prediction Tool (Depends on Cell 8)
# ============================================================================
@app.cell
def _(ui, mo, model, scaler, predicted_avg_price):
    """
    ## Interactive Price Predictor
    
    This cell creates an interactive tool to predict property prices
    based on user inputs. It depends on the trained model from Cell 8.
    """
    # Create input widgets for prediction
    sqft_input = ui.number(
        start=800,
        stop=4000,
        step=50,
        value=2000,
        label="Square Footage"
    )
    
    bedrooms_input = ui.number(
        start=2,
        stop=5,
        step=1,
        value=3,
        label="Bedrooms"
    )
    
    bathrooms_input = ui.number(
        start=1,
        stop=3,
        step=0.5,
        value=2,
        label="Bathrooms"
    )
    
    age_input = ui.number(
        start=0,
        stop=100,
        step=5,
        value=20,
        label="Property Age (years)"
    )
    
    location_input = ui.number(
        start=0,
        stop=10,
        step=0.5,
        value=5,
        label="Location Score (0-10)"
    )
    
    # Get current values for prediction
    current_values = [
        sqft_input.value,
        bedrooms_input.value,
        bathrooms_input.value,
        age_input.value,
        location_input.value
    ]
    
    # Prepare input for model
    input_array = np.array(current_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    
    # Make prediction
    predicted_price = model.predict(input_scaled)[0]
    price_difference = predicted_price - predicted_avg_price
    
    return mo.md(f"""
    ### 🎯 Interactive Price Predictor
    
    Adjust the values below to predict property prices:
    
    **Input Parameters:**
    - {sqft_input}
    - {bedrooms_input}
    - {bathrooms_input}
    - {age_input}
    - {location_input}
    
    **Prediction Results:**
    
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; 
                border-radius: 10px; 
                color: white; 
                margin: 20px 0;">
        <h3 style="color: white; margin: 0;">
        Predicted Price: ${predicted_price:,.0f}
        </h3>
        <p style="margin: 10px 0 0 0;">
        This is ${abs(price_difference):,.0f} {'above' if price_difference > 0 else 'below'} the average filtered property
        </p>
    </div>
    
    **Current Inputs:**
    - Square Footage: {current_values[0]:.0f}
    - Bedrooms: {current_values[1]:.0f}
    - Bathrooms: {current_values[2]:.1f}
    - Age: {current_values[3]:.0f} years
    - Location Score: {current_values[4]:.1f}/10
    
    *Predictor created by: 23f2001189@ds.study.iitm.ac.in*
    """), sqft_input, bedrooms_input, bathrooms_input, age_input, location_input, predicted_price, price_difference


# ============================================================================
# CELL 10: Summary and Conclusions
# ============================================================================
@app.cell
def _(mo, count, avg_price, strongest_corr_feature, strongest_corr_value, r2, rmse):
    """
    ## Analysis Summary
    
    This cell provides a summary of the entire analysis.
    It depends on results from multiple previous cells.
    """
    return mo.md(f"""
    # 📊 Analysis Summary
    
    ## Key Findings
    
    1. **Dataset Overview:**
       - Analyzed {count} properties after filtering
       - Average price: ${avg_price:,.0f}
    
    2. **Strongest Relationship:**
       - {strongest_corr_feature.replace('_', ' ').title()} shows the strongest correlation with price
       - Correlation coefficient: {strongest_corr_value:.3f}
    
    3. **Model Performance:**
       - Linear regression achieved R² = {r2:.4f}
       - Prediction error (RMSE): ${rmse:,.0f}
    
    4. **Interactive Features:**
       - Real-time filtering with sliders
       - Dynamic visualizations
       - Interactive price predictor
    
    ## Recommendations
    
    1. **For Buyers:** Focus on {strongest_corr_feature.replace('_', ' ')} as it has the strongest impact on price
    2. **For Sellers:** Consider renovations that improve the {strongest_corr_feature.replace('_', ' ')}
    3. **For Investors:** Use the interactive predictor to estimate property values
    
    ## Contact Information
    
    **Data Scientist:** 23f2001189@ds.study.iitm.ac.in
    
    This analysis demonstrates the power of interactive notebooks for
    exploratory data analysis and stakeholder communication.
    
    ---
    
    *Notebook generated using Marimo - Reactive Python Notebooks*
    *Last updated: December 2023*
    """)


# ============================================================================
# Run the app
# ============================================================================
if __name__ == "__main__":
    app.run()
