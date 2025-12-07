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
# CELL 4: INTERACTIVE SLIDER WIDGETS SECTION
# ============================================================================
@app.cell
def _(ui, mo):
    """
    ## 🎛️ Interactive Control Panel
    
    This cell creates multiple interactive slider widgets to control
    the data analysis in real-time. All widgets update dependent cells.
    
    Contact: 23f2001189@ds.study.iitm.ac.in
    """
    # === INTERACTIVE SLIDER WIDGET 1: Price Range Filter ===
    price_slider = ui.slider(
        start=150000,
        stop=1500000,
        step=50000,
        value=[300000, 800000],
        label="💰 Price Range ($)",
        thumb_label=True
    )
    
    # === INTERACTIVE SLIDER WIDGET 2: Bedroom Filter ===
    bedroom_slider = ui.slider(
        start=2,
        stop=5,
        step=1,
        value=3,
        label="🛏️ Number of Bedrooms",
        thumb_label=True
    )
    
    # === INTERACTIVE SLIDER WIDGET 3: Age Filter ===
    age_slider = ui.slider(
        start=0,
        stop=100,
        step=5,
        value=[10, 30],
        label="🏠 Property Age Range (years)",
        thumb_label=True
    )
    
    # === INTERACTIVE SLIDER WIDGET 4: Square Footage Filter ===
    sqft_slider = ui.slider(
        start=800,
        stop=4000,
        step=100,
        value=[1500, 2500],
        label="📐 Square Footage Range",
        thumb_label=True
    )
    
    # === INTERACTIVE SLIDER WIDGET 5: Location Score ===
    location_slider = ui.slider(
        start=0,
        stop=10,
        step=0.5,
        value=5.0,
        label="📍 Minimum Location Score (0-10)",
        thumb_label=True
    )
    
    # === INTERACTIVE SLIDER WIDGET 6: Correlation Threshold ===
    correlation_slider = ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.3,
        label="📊 Minimum Correlation Threshold",
        thumb_label=True
    )
    
    # Display the widgets in a nice layout
    return mo.md(f"""
    ## 🎚️ Interactive Analysis Controls
    
    Adjust these sliders to interactively filter and analyze the housing data:
    
    ### Filter Controls
    {price_slider}
    
    {bedroom_slider}
    
    {age_slider}
    
    {sqft_slider}
    
    {location_slider}
    
    ### Analysis Controls
    {correlation_slider}
    
    *Note: All visualizations and statistics below will update automatically when you adjust these sliders.*
    
    **Analyst:** 23f2001189@ds.study.iitm.ac.in
    """), price_slider, bedroom_slider, age_slider, sqft_slider, location_slider, correlation_slider


# ============================================================================
# CELL 5: Filter Data Based on Widgets (Depends on Cells 2 and 4)
# ============================================================================
@app.cell
def _(data, price_slider, bedroom_slider, age_slider, sqft_slider, location_slider, mo):
    """
    ## Apply Filters and Display Results
    
    This cell filters the dataset based on ALL slider values from Cell 4.
    It demonstrates variable dependencies between cells.
    """
    # Get current slider values
    min_price, max_price = price_slider.value
    selected_bedrooms = bedroom_slider.value
    min_age, max_age = age_slider.value
    min_sqft, max_sqft = sqft_slider.value
    min_location = location_slider.value
    
    # Apply ALL filters
    filtered_data = data[
        (data['price'] >= min_price) &
        (data['price'] <= max_price) &
        (data['bedrooms'] == selected_bedrooms) &
        (data['age'] >= min_age) &
        (data['age'] <= max_age) &
        (data['square_footage'] >= min_sqft) &
        (data['square_footage'] <= max_sqft) &
        (data['location_score'] >= min_location)
    ]
    
    # Calculate statistics for filtered data
    avg_price = filtered_data['price'].mean()
    avg_sqft = filtered_data['square_footage'].mean()
    avg_age = filtered_data['age'].mean()
    avg_location = filtered_data['location_score'].mean()
    count = len(filtered_data)
    
    # Calculate percentage of total data
    total_count = len(data)
    percentage = (count / total_count * 100) if total_count > 0 else 0
    
    # Dynamic markdown output based on widget state
    return mo.md(f"""
    ### 🔍 Filtered Results
    
    **Applied Filters:**
    - Price Range: ${min_price:,} - ${max_price:,}
    - Bedrooms: {selected_bedrooms}
    - Age Range: {min_age} - {max_age} years
    - Square Footage: {min_sqft:,} - {max_sqft:,} sq ft
    - Minimum Location Score: {min_location}/10
    
    **Results:**
    - 📊 Properties Found: **{count}** ({percentage:.1f}% of total)
    - 💰 Average Price: **${avg_price:,.0f}**
    - 📐 Average Square Footage: **{avg_sqft:.0f} sq ft**
    - 🏠 Average Age: **{avg_age:.1f} years**
    - 📍 Average Location Score: **{avg_location:.1f}/10**
    
    **Price Metrics:**
    - Price per Sq Ft: **${(avg_price/avg_sqft):.0f}**
    - Minimum Price: **${filtered_data['price'].min():,.0f}**
    - Maximum Price: **${filtered_data['price'].max():,.0f}**
    
    *Analysis by: 23f2001189@ds.study.iitm.ac.in*
    
    ---
    
    *Try adjusting the sliders above to see how the results change in real-time!*
    """), filtered_data, min_price, max_price, selected_bedrooms, min_age, max_age, min_sqft, max_sqft, min_location, avg_price, avg_sqft, avg_age, avg_location, count, percentage


# ============================================================================
# CELL 6: Correlation Analysis with Threshold (Depends on Cells 4 and 5)
# ============================================================================
@app.cell
def _(filtered_data, correlation_slider, mo, np):
    """
    ## Correlation Analysis with Interactive Threshold
    
    This cell calculates correlations between variables with an interactive
    threshold filter from the slider. It depends on 'filtered_data' from Cell 5
    and 'correlation_slider' from Cell 4.
    """
    # Get current correlation threshold
    corr_threshold = correlation_slider.value
    
    # Calculate correlation matrix
    corr_matrix = filtered_data.corr()
    
    # Calculate individual correlations with price
    price_correlations = {}
    for col in filtered_data.columns:
        if col != 'price':
            corr = np.corrcoef(filtered_data['price'], filtered_data[col])[0, 1]
            price_correlations[col] = corr
    
    # Filter correlations based on threshold
    significant_correlations = {
        feature: corr 
        for feature, corr in price_correlations.items() 
        if abs(corr) >= corr_threshold
    }
    
    # Find strongest correlation
    if price_correlations:
        strongest_corr_feature = max(price_correlations, key=lambda k: abs(price_correlations[k]))
        strongest_corr_value = price_correlations[strongest_corr_feature]
    else:
        strongest_corr_feature = "N/A"
        strongest_corr_value = 0
    
    # Create correlation table
    corr_rows = []
    for feature, corr in price_correlations.items():
        significance = "✅ Significant" if abs(corr) >= corr_threshold else "⚠️ Below threshold"
        direction = "📈 Positive" if corr > 0 else "📉 Negative"
        strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.3 else "Weak"
        
        corr_rows.append(f"""
        <tr>
            <td>{feature.replace('_', ' ').title()}</td>
            <td>{corr:.3f}</td>
            <td>{direction}</td>
            <td>{strength}</td>
            <td>{significance}</td>
        </tr>
        """)
    
    return mo.md(f"""
    ### 📊 Interactive Correlation Analysis
    
    **Current Threshold:** {corr_threshold:.2f}
    
    **Correlations with Price:**
    
    <table style="width:100%">
        <tr>
            <th>Feature</th>
            <th>Correlation</th>
            <th>Direction</th>
            <th>Strength</th>
            <th>Significance</th>
        </tr>
        {''.join(corr_rows)}
    </table>
    
    **Key Findings:**
    - Strongest Relationship: **{strongest_corr_feature.replace('_', ' ').title()}** 
      (r = {strongest_corr_value:.3f})
    - Significant Correlations: **{len(significant_correlations)}** out of {len(price_correlations)} features
    - Threshold Filter: Showing correlations ≥ {corr_threshold:.2f} in absolute value
    
    **Interpretation Guide:**
    - |r| ≥ 0.7: Strong relationship
    - 0.3 ≤ |r| < 0.7: Moderate relationship
    - |r| < 0.3: Weak relationship
    
    *Adjust the correlation threshold slider to filter relationships!*
    
    *Contact: 23f2001189@ds.study.iitm.ac.in*
    """), corr_matrix, price_correlations, significant_correlations, strongest_corr_feature, strongest_corr_value, corr_threshold


# ============================================================================
# CELL 7: Visualizations with Interactive Updates (Depends on Cells 4, 5, 6)
# ============================================================================
@app.cell
def _(filtered_data, plt, sns, price_correlations, strongest_corr_feature, corr_threshold, mo):
    """
    ## Interactive Data Visualization
    
    This cell creates visualizations that update based on slider inputs.
    It depends on multiple variables from previous cells.
    """
    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 12))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Price distribution with current filter range
    axes[0, 0].hist(filtered_data['price'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].axvline(filtered_data['price'].mean(), color='red', linestyle='--', 
                      label=f'Mean: ${filtered_data["price"].mean():,.0f}')
    axes[0, 0].set_xlabel('Price ($)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Price Distribution with Current Filters', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Price vs strongest correlated feature
    if strongest_corr_feature != "N/A":
        axes[0, 1].scatter(filtered_data[strongest_corr_feature], 
                          filtered_data['price'], 
                          alpha=0.6, 
                          edgecolors='w', 
                          s=50,
                          c=filtered_data['location_score'],
                          cmap='viridis')
        axes[0, 1].set_xlabel(strongest_corr_feature.replace('_', ' ').title(), fontsize=11)
        axes[0, 1].set_ylabel('Price ($)', fontsize=11)
        axes[0, 1].set_title(f'Price vs {strongest_corr_feature.replace("_", " ").title()}\n(r = {price_correlations.get(strongest_corr_feature, 0):.3f})', 
                           fontsize=13, fontweight='bold')
        
        # Add regression line
        z = np.polyfit(filtered_data[strongest_corr_feature], filtered_data['price'], 1)
        p = np.poly1d(z)
        axes[0, 1].plot(filtered_data[strongest_corr_feature], 
                       p(filtered_data[strongest_corr_feature]), 
                       "r--", 
                       alpha=0.8,
                       linewidth=2,
                       label='Regression Line')
        axes[0, 1].legend()
        
        # Add colorbar for location score
        plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1], label='Location Score')
    
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Correlation heatmap
    corr_matrix = filtered_data.corr()
    im = axes[1, 0].imshow(corr_matrix, cmap='RdYlBu', aspect='auto', vmin=-1, vmax=1)
    axes[1, 0].set_title(f'Correlation Matrix\n(Threshold ≥ {corr_threshold:.2f})', 
                        fontsize=13, fontweight='bold')
    axes[1, 0].set_xticks(range(len(corr_matrix.columns)))
    axes[1, 0].set_yticks(range(len(corr_matrix.columns)))
    axes[1, 0].set_xticklabels([col.replace('_', '\n').title() for col in corr_matrix.columns], 
                              rotation=45, ha='right', fontsize=9)
    axes[1, 0].set_yticklabels([col.replace('_', '\n').title() for col in corr_matrix.columns], 
                              fontsize=9)
    
    # Add correlation values (only show significant ones)
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= corr_threshold:
                color = "white" if abs(corr_value) > 0.5 else "black"
                axes[1, 0].text(j, i, f'{corr_value:.2f}',
                              ha="center", va="center", 
                              color=color, fontsize=10, fontweight='bold')
    
    # Add colorbar
    plt.colorbar(im, ax=axes[1, 0], label='Correlation Coefficient')
    
    # Plot 4: Feature importance (absolute correlation) with threshold line
    features = list(price_correlations.keys())
    corr_values = [abs(price_correlations[f]) for f in features]
    
    y_pos = np.arange(len(features))
    bars = axes[1, 1].barh(y_pos, corr_values, alpha=0.7, 
                          color=['green' if v >= corr_threshold else 'gray' for v in corr_values])
    axes[1, 1].axvline(x=corr_threshold, color='red', linestyle='--', 
                      label=f'Threshold: {corr_threshold:.2f}')
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels([f.replace('_', ' ').title() for f in features])
    axes[1, 1].set_xlabel('Absolute Correlation with Price', fontsize=11)
    axes[1, 1].set_title('Feature Importance\n(Colored by significance)', 
                        fontsize=13, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, corr_values)):
        width = bar.get_width()
        axes[1, 1].text(width + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{value:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    return mo.md(f"""
    ### 📈 Interactive Visualizations
    
    These plots update automatically based on your filter selections:
    
    1. **Price Distribution** - Shows filtered property prices with mean line
    2. **Price vs {strongest_corr_feature.replace('_', ' ').title() if strongest_corr_feature != "N/A" else "Key Feature"}** - Scatter plot with regression line (colored by location)
    3. **Correlation Matrix** - Heatmap showing relationships (values ≥ {corr_threshold:.2f} shown)
    4. **Feature Importance** - Bar chart showing absolute correlations (green = significant)
    
    *Try adjusting the correlation threshold slider to see how the heatmap and bar chart change!*
    
    *Visualizations generated by: 23f2001189@ds.study.iitm.ac.in*
    """), fig


# ============================================================================
# CELL 8: Statistical Model with Interactive Features (Depends on Cells 4, 5)
# ============================================================================
@app.cell
def _(filtered_data, corr_threshold, mo):
    """
    ## Statistical Modeling with Interactive Threshold
    
    This cell performs linear regression based on filtered data and threshold.
    It depends on the filtered dataset from Cell 5 and threshold from Cell 4.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    
    # Only use features with correlation above threshold
    corr_with_price = filtered_data.corr()['price'].abs()
    significant_features = corr_with_price[corr_with_price >= corr_threshold].index.tolist()
    
    # Remove 'price' from features if present
    if 'price' in significant_features:
        significant_features.remove('price')
    
    if len(significant_features) == 0:
        significant_features = filtered_data.columns.drop('price').tolist()
    
    # Prepare features and target
    X = filtered_data[significant_features]
    y = filtered_data['price']
    
    # Check if we have enough data
    if len(X) < 2:
        return mo.md("""
        ### ⚠️ Insufficient Data for Modeling
        
        The current filters result in too few data points for statistical modeling.
        Please adjust the sliders to include more properties.
        
        *Contact: 23f2001189@ds.study.iitm.ac.in*
        """), None, None, None, None, None, None, None, None
    
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
    
    # Calculate coefficients and importance
    feature_names = X.columns
    coefficients = model.coef_
    intercept = model.intercept_
    
    # Create coefficient table sorted by absolute impact
    coeff_table = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Absolute_Impact': np.abs(coefficients),
        'Standardized_Coefficient': coefficients / np.std(X_scaled, axis=0) if len(X_scaled) > 0 else coefficients
    }).sort_values('Absolute_Impact', ascending=False)
    
    # Calculate price prediction for average property
    if len(X) > 0:
        avg_property = X.mean().values.reshape(1, -1)
        avg_property_scaled = scaler.transform(avg_property)
        predicted_avg_price = model.predict(avg_property_scaled)[0]
    else:
        predicted_avg_price = 0
    
    return mo.md(f"""
    ### 🧮 Interactive Linear Regression Model
    
    **Model Details:**
    - Features used: **{len(significant_features)}** (correlation ≥ {corr_threshold:.2f})
    - Samples: **{len(X)}** properties
    - R² Score: **{r2:.4f}**
    - RMSE: **${rmse:,.0f}**
    
    **Feature Coefficients:**
    
    | Feature | Coefficient | Impact per Std Dev |
    |---------|-------------|-------------------|
    {''.join([f"| {row['Feature'].replace('_', ' ').title()} | ${row['Coefficient']:,.0f} | ${row['Standardized_Coefficient']:,.0f} |\n" for _, row in coeff_table.iterrows()])}
    
    **Model Equation:**
    Price = ${intercept:,.0f} {' + '.join([f'({coeff:,.0f} × {feat})' for feat, coeff in zip(feature_names, coefficients)])}
    
    **Interpretation:**
    - Intercept: Base price of **${intercept:,.0f}**
    - Most important feature: **{coeff_table.iloc[0]['Feature'].replace('_', ' ').title()}**
    - Each standard deviation increase adds **${abs(coeff_table.iloc[0]['Standardized_Coefficient']):,.0f}** to price
    
    **Predicted Price for Average Property:** ${predicted_avg_price:,.0f}
    
    *Model automatically updates based on correlation threshold!*
    
    *Model developed by: 23f2001189@ds.study.iitm.ac.in*
    """), model, r2, rmse, coefficients, intercept, coeff_table, predicted_avg_price, scaler, significant_features


# ============================================================================
# CELL 9: Interactive Prediction Tool (Depends on Cell 8)
# ============================================================================
@app.cell
def _(ui, mo, model, scaler, significant_features, predicted_avg_price):
    """
    ## Interactive Price Predictor
    
    This cell creates an interactive tool to predict property prices
    based on user inputs. It depends on the trained model from Cell 8.
    """
    # Create dynamic input widgets based on significant features
    input_widgets = {}
    input_values = {}
    
    # Default ranges for features
    feature_ranges = {
        'square_footage': (800, 4000, 2000),
        'bedrooms': (2, 5, 3),
        'bathrooms': (1, 3, 2),
        'age': (0, 100, 20),
        'location_score': (0, 10, 5)
    }
    
    # Create widgets only for significant features
    widget_html = []
    for feature in significant_features:
        if feature in feature_ranges:
            start, stop, default = feature_ranges[feature]
            
            if feature in ['bedrooms', 'bathrooms']:
                widget = ui.slider(
                    start=start,
                    stop=stop,
                    step=0.5 if feature == 'bathrooms' else 1,
                    value=default,
                    label=f"{feature.replace('_', ' ').title()}",
                    thumb_label=True
                )
            else:
                widget = ui.slider(
                    start=start,
                    stop=stop,
                    step=(stop-start)/100,
                    value=default,
                    label=f"{feature.replace('_', ' ').title()}",
                    thumb_label=True
                )
            
            input_widgets[feature] = widget
            input_values[feature] = widget.value
            widget_html.append(f"{widget}")
    
    # Prepare input for model
    if significant_features and input_values:
        input_array = np.array([input_values[f] for f in significant_features]).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        predicted_price = model.predict(input_scaled)[0]
        price_difference = predicted_price - predicted_avg_price
    else:
        predicted_price = 0
        price_difference = 0
    
    return mo.md(f"""
    ### 🎯 Interactive Price Predictor
    
    Adjust the sliders below to predict property prices in real-time:
    
    **Input Parameters:**
    {''.join(widget_html)}
    
    **Current Inputs:**
    {''.join([f"- {feat.replace('_', ' ').title()}: {val:.1f}<br>" for feat, val in input_values.items()])}
    
    **Prediction Results:**
    
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 25px; 
                border-radius: 15px; 
                color: white; 
                margin: 25px 0;
                text-align: center;
                box-shadow: 0 10px 20px rgba(0,0,0,0.2);">
        <h2 style="color: white; margin: 0; font-size: 2em;">
        ${predicted_price:,.0f}
        </h2>
        <p style="margin: 15px 0 0 0; font-size: 1.2em;">
        Estimated Property Value
        </p>
        <p style="margin: 10px 0 0 0; opacity: 0.9;">
        {f'${abs(price_difference):,.0f} {"above" if price_difference > 0 else "below"} average'}
        </p>
    </div>
    
    **Features Used in Prediction:**
    - Model uses {len(significant_features)} features with correlation above threshold
    - Each slider adjusts one feature in the prediction model
    
    *Try adjusting the sliders to see how each feature affects the predicted price!*
    
    *Predictor created by: 23f2001189@ds.study.iitm.ac.in*
    """), input_widgets, input_values, predicted_price, price_difference


# ============================================================================
# CELL 10: Summary Dashboard (Depends on Multiple Cells)
# ============================================================================
@app.cell
def _(mo, count, percentage, avg_price, strongest_corr_feature, strongest_corr_value, 
      corr_threshold, r2, rmse, len, significant_features):
    """
    ## Interactive Analysis Dashboard
    
    This cell provides a comprehensive summary that updates with all interactions.
    It depends on results from multiple previous cells.
    """
    return mo.md(f"""
    # 📊 Interactive Analysis Dashboard
    
    ## 🎛️ Current Settings Summary
    
    **Data Filters:**
    - Properties analyzed: **{count}** ({percentage:.1f}% of total dataset)
    - Average filtered price: **${avg_price:,.0f}**
    
    **Analysis Parameters:**
    - Correlation threshold: **{corr_threshold:.2f}**
    - Significant features in model: **{len(significant_features)}**
    
    ## 📈 Key Insights
    
    1. **Strongest Relationship Found:**
       - Feature: **{strongest_corr_feature.replace('_', ' ').title() if strongest_corr_feature != "N/A" else "N/A"}**
       - Correlation: **{strongest_corr_value:.3f}**
       - This feature explains the most variation in housing prices
    
    2. **Model Performance:**
       - Prediction accuracy (R²): **{r2:.4f}**
       - Average prediction error: **${rmse:,.0f}**
    
    3. **Interactive Features:**
       - 6 interactive sliders controlling filters and analysis
       - Real-time visual updates
       - Dynamic statistical modeling
       - Live price prediction tool
    
    ## 🔄 How to Use This Notebook
    
    1. **Adjust filters** using the sliders in the control panel
    2. **Observe changes** in visualizations and statistics
    3. **Modify analysis parameters** like correlation threshold
    4. **Use the predictor** to estimate property values
    5. **All components update automatically** in real-time
    
    ## 📧 Contact & Information
    
    **Data Scientist:** 23f2001189@ds.study.iitm.ac.in
    
    **Notebook Features:**
    - Reactive programming with Marimo
    - Real-time data filtering
    - Interactive visualizations
    - Statistical modeling
    - Predictive analytics
    
    ---
    
    *This interactive notebook demonstrates the power of reactive data analysis.
    All components are connected - changing any slider updates the entire analysis.*
    
    *Last updated: December 2023 | Version: 2.0*
    """)


# ============================================================================
# Run the app
# ============================================================================
if __name__ == "__main__":
    app.run()
