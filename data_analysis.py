"""
# Interactive Data Analysis Notebook
## Research Institution - Data Science Division

**Analyst:** 23f2001189@ds.study.iitm.ac.in
**Dataset:** Housing Price Analysis
**Created:** December 2023
"""

import marimo

__generated_with = "0.1.64"
app = marimo.App(width="full", layout_file="layouts/app.py")


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
def _(pd, np):
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
# CELL 4: INTERACTIVE SLIDER WIDGET - PRICE RANGE FILTER
# ============================================================================
@app.cell
def _(ui, mo):
    """
    ## 🎛️ INTERACTIVE SLIDER WIDGET: Price Range Filter
    
    This cell creates an interactive slider widget to filter properties by price.
    The slider allows real-time interaction and updates all dependent cells.
    
    Analyst: 23f2001189@ds.study.iitm.ac.in
    """
    # Create the interactive slider widget
    price_slider = ui.slider(
        start=150000,      # Minimum price
        stop=1500000,      # Maximum price
        step=50000,        # Increment step
        value=[300000, 800000],  # Default range
        label="💰 Adjust Price Range ($)",
        thumb_label=True,   # Show current values on thumbs
        full_width=True     # Use full width
    )
    
    # Display the slider with instructions
    return mo.md(f"""
    ### 🎚️ Interactive Price Filter
    
    **Use the slider below to filter properties by price range:**
    
    {price_slider}
    
    **Current Selection:** ${price_slider.value[0]:,} - ${price_slider.value[1]:,}
    
    **How it works:**
    1. Move the left handle to set minimum price
    2. Move the right handle to set maximum price
    3. All charts and statistics below will update automatically
    
    *Try sliding to see real-time updates!*
    
    **Analyst Contact:** 23f2001189@ds.study.iitm.ac.in
    """), price_slider


# ============================================================================
# CELL 5: INTERACTIVE SLIDER WIDGET - CORRELATION THRESHOLD
# ============================================================================
@app.cell
def _(ui, mo):
    """
    ## 🎛️ INTERACTIVE SLIDER WIDGET: Correlation Threshold
    
    This cell creates a second interactive slider widget to control
    the correlation threshold for statistical analysis.
    
    Analyst: 23f2001189@ds.study.iitm.ac.in
    """
    # Create correlation threshold slider
    correlation_slider = ui.slider(
        start=0.0,        # Minimum correlation
        stop=1.0,         # Maximum correlation
        step=0.05,        # Increment step
        value=0.3,        # Default value
        label="📊 Correlation Threshold (absolute value)",
        thumb_label=True,  # Show current value
        full_width=True    # Use full width
    )
    
    # Display the slider
    return mo.md(f"""
    ### 🎚️ Interactive Correlation Control
    
    **Use this slider to adjust the statistical significance threshold:**
    
    {correlation_slider}
    
    **Current Threshold:** {correlation_slider.value}
    
    **Interpretation:**
    - Only correlations with absolute value ≥ threshold will be considered significant
    - Lower values show more relationships
    - Higher values show only strong relationships
    
    **Correlation Strength Guide:**
    - 0.0-0.3: Weak correlation
    - 0.3-0.7: Moderate correlation  
    - 0.7-1.0: Strong correlation
    
    *Adjust to explore different levels of relationship strength!*
    
    **Analyst:** 23f2001189@ds.study.iitm.ac.in
    """), correlation_slider


# ============================================================================
# CELL 6: INTERACTIVE SLIDER WIDGET - PROPERTY AGE
# ============================================================================
@app.cell
def _(ui, mo):
    """
    ## 🎛️ INTERACTIVE SLIDER WIDGET: Property Age Filter
    
    This cell creates a third interactive slider widget to filter by property age.
    Demonstrates range selection with two handles.
    
    Analyst: 23f2001189@ds.study.iitm.ac.in
    """
    # Create age range slider
    age_slider = ui.slider(
        start=0,          # Minimum age
        stop=100,         # Maximum age
        step=5,           # Increment step
        value=[10, 30],   # Default range
        label="🏠 Property Age Range (years)",
        thumb_label=True, # Show current values
        full_width=True   # Use full width
    )
    
    return mo.md(f"""
    ### 🎚️ Interactive Age Filter
    
    **Filter properties by age range:**
    
    {age_slider}
    
    **Current Selection:** {age_slider.value[0]} - {age_slider.value[1]} years
    
    **Age Categories:**
    - 0-10 years: New construction
    - 10-30 years: Established properties
    - 30+ years: Older properties
    
    *Move both handles to select your desired age range!*
    
    **Contact:** 23f2001189@ds.study.iitm.ac.in
    """), age_slider


# ============================================================================
# CELL 7: Filter Data Based on Interactive Sliders (Depends on Cells 2, 4, 6)
# ============================================================================
@app.cell
def _(data, price_slider, age_slider, mo):
    """
    ## Apply Interactive Filters
    
    This cell filters the dataset based on slider values from Cells 4 and 6.
    Demonstrates variable dependencies between cells.
    """
    # Get current slider values
    min_price, max_price = price_slider.value
    min_age, max_age = age_slider.value
    
    # Apply filters
    filtered_data = data[
        (data['price'] >= min_price) &
        (data['price'] <= max_price) &
        (data['age'] >= min_age) &
        (data['age'] <= max_age)
    ]
    
    # Calculate statistics
    avg_price = filtered_data['price'].mean()
    avg_age = filtered_data['age'].mean()
    count = len(filtered_data)
    percentage = (count / len(data) * 100) if len(data) > 0 else 0
    
    # Dynamic markdown output based on slider states
    return mo.md(f"""
    ### 🔍 Interactive Filter Results
    
    **Current Filters Applied:**
    - Price Range: **${min_price:,} - ${max_price:,}**
    - Age Range: **{min_age} - {max_age} years**
    
    **Results:**
    - 📊 Properties Found: **{count}** ({percentage:.1f}% of total)
    - 💰 Average Price: **${avg_price:,.0f}**
    - 🏠 Average Age: **{avg_age:.1f} years**
    - 📈 Price Range: **${filtered_data['price'].min():,} - ${filtered_data['price'].max():,}**
    
    **Price Distribution:**
    - 25th Percentile: **${filtered_data['price'].quantile(0.25):,.0f}**
    - Median: **${filtered_data['price'].median():,.0f}**
    - 75th Percentile: **${filtered_data['price'].quantile(0.75):,.0f}**
    
    *Try adjusting the sliders above to see real-time changes in these statistics!*
    
    **Analysis by:** 23f2001189@ds.study.iitm.ac.in
    """), filtered_data, min_price, max_price, min_age, max_age, avg_price, avg_age, count, percentage


# ============================================================================
# CELL 8: Correlation Analysis with Interactive Threshold (Depends on Cells 5, 7)
# ============================================================================
@app.cell
def _(filtered_data, correlation_slider, mo, np):
    """
    ## Interactive Correlation Analysis
    
    This cell performs correlation analysis using the threshold from Cell 5.
    Shows variable dependencies between cells.
    """
    # Get current threshold from slider
    threshold = correlation_slider.value
    
    # Calculate correlations
    correlations = {}
    for col in filtered_data.columns:
        if col != 'price':
            corr = np.corrcoef(filtered_data['price'], filtered_data[col])[0, 1]
            correlations[col] = corr
    
    # Find significant correlations
    significant = {k: v for k, v in correlations.items() if abs(v) >= threshold}
    
    # Find strongest correlation
    if correlations:
        strongest_feature = max(correlations, key=lambda k: abs(correlations[k]))
        strongest_value = correlations[strongest_feature]
    else:
        strongest_feature = "N/A"
        strongest_value = 0
    
    # Create correlation table
    rows = []
    for feature, corr in correlations.items():
        is_significant = abs(corr) >= threshold
        significance = "✅ Significant" if is_significant else "⚠️ Below threshold"
        strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.3 else "Weak"
        direction = "📈 Positive" if corr > 0 else "📉 Negative"
        
        rows.append(f"""
        <tr style="background-color: {'#e8f5e8' if is_significant else '#f5f5f5'}">
            <td>{feature.replace('_', ' ').title()}</td>
            <td>{corr:.3f}</td>
            <td>{strength}</td>
            <td>{direction}</td>
            <td>{significance}</td>
        </tr>
        """)
    
    return mo.md(f"""
    ### 📊 Interactive Correlation Analysis
    
    **Current Threshold:** {threshold:.2f}
    **Significant Correlations:** {len(significant)} of {len(correlations)}
    
    **Correlation Matrix:**
    
    <table style="width:100%; border-collapse: collapse;">
        <tr style="background-color: #2c3e50; color: white;">
            <th>Feature</th>
            <th>Correlation</th>
            <th>Strength</th>
            <th>Direction</th>
            <th>Significance</th>
        </tr>
        {''.join(rows)}
    </table>
    
    **Key Finding:**
    - Strongest relationship: **{strongest_feature.replace('_', ' ').title()}** (r = {strongest_value:.3f})
    
    **How to use:**
    1. Adjust the correlation threshold slider above
    2. Watch the table update in real-time
    3. Green rows show significant correlations
    4. Try setting threshold to 0.5 to see only strong relationships
    
    *Analysis Contact: 23f2001189@ds.study.iitm.ac.in*
    """), correlations, significant, strongest_feature, strongest_value, threshold


# ============================================================================
# CELL 9: Interactive Visualization (Depends on Cells 4, 5, 6, 7, 8)
# ============================================================================
@app.cell
def _(filtered_data, plt, sns, correlations, strongest_feature, threshold, mo):
    """
    ## Interactive Visualization Dashboard
    
    This cell creates visualizations that update with slider interactions.
    Shows complex dependencies between multiple cells.
    """
    # Set up the plot
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Price distribution histogram
    axes[0, 0].hist(filtered_data['price'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(filtered_data['price'].mean(), color='red', linestyle='--', 
                      label=f'Mean: ${filtered_data["price"].mean():,.0f}')
    axes[0, 0].set_xlabel('Price ($)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Price Distribution (Filtered)', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot of strongest correlation
    if strongest_feature != "N/A" and strongest_feature in filtered_data.columns:
        scatter = axes[0, 1].scatter(
            filtered_data[strongest_feature], 
            filtered_data['price'],
            c=filtered_data['age'],
            cmap='viridis',
            alpha=0.7,
            s=50,
            edgecolor='white'
        )
        axes[0, 1].set_xlabel(strongest_feature.replace('_', ' ').title(), fontsize=11)
        axes[0, 1].set_ylabel('Price ($)', fontsize=11)
        axes[0, 1].set_title(f'Price vs {strongest_feature.replace("_", " ").title()}\nr = {correlations.get(strongest_feature, 0):.3f}', 
                           fontsize=13, fontweight='bold')
        
        # Add colorbar for age
        plt.colorbar(scatter, ax=axes[0, 1], label='Property Age')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Correlation heatmap
    corr_matrix = filtered_data.corr()
    im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    axes[1, 0].set_title(f'Correlation Matrix\n(Threshold ≥ {threshold:.2f})', fontsize=13, fontweight='bold')
    axes[1, 0].set_xticks(range(len(corr_matrix.columns)))
    axes[1, 0].set_yticks(range(len(corr_matrix.columns)))
    axes[1, 0].set_xticklabels([col.replace('_', '\n').title() for col in corr_matrix.columns], 
                              rotation=45, ha='right', fontsize=9)
    axes[1, 0].set_yticklabels([col.replace('_', '\n').title() for col in corr_matrix.columns], 
                              fontsize=9)
    
    # Add correlation values (only significant ones)
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if i != j and abs(corr_value) >= threshold:
                color = "white" if abs(corr_value) > 0.5 else "black"
                axes[1, 0].text(j, i, f'{corr_value:.2f}',
                              ha="center", va="center", 
                              color=color, fontsize=9, fontweight='bold')
    
    plt.colorbar(im, ax=axes[1, 0], label='Correlation Coefficient')
    
    # Plot 4: Bar chart of correlations
    features = list(correlations.keys())
    corr_values = [correlations[f] for f in features]
    abs_values = [abs(v) for v in corr_values]
    
    y_pos = np.arange(len(features))
    colors = ['green' if abs(v) >= threshold else 'gray' for v in corr_values]
    
    bars = axes[1, 1].barh(y_pos, corr_values, color=colors, alpha=0.7)
    axes[1, 1].axvline(x=0, color='black', linewidth=0.8)
    axes[1, 1].axvline(x=threshold, color='red', linestyle='--', alpha=0.5, label=f'Threshold: {threshold:.2f}')
    axes[1, 1].axvline(x=-threshold, color='red', linestyle='--', alpha=0.5)
    
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels([f.replace('_', ' ').title() for f in features])
    axes[1, 1].set_xlabel('Correlation with Price', fontsize=11)
    axes[1, 1].set_title('Feature Correlations (Green = Significant)', fontsize=13, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, corr_values)):
        width = bar.get_width()
        axes[1, 1].text(width + (0.02 if width >= 0 else -0.02), 
                       bar.get_y() + bar.get_height()/2,
                       f'{value:.3f}', 
                       ha='left' if width >= 0 else 'right', 
                       va='center', 
                       fontsize=9)
    
    plt.tight_layout()
    
    return mo.md(f"""
    ### 📈 Interactive Visualizations
    
    **All plots update automatically when you adjust the sliders!**
    
    **Current View:**
    1. **Price Distribution** - Shows filtered property prices
    2. **Scatter Plot** - Price vs strongest correlated feature (colored by age)
    3. **Correlation Matrix** - Heatmap of all correlations (values ≥ {threshold:.2f} shown)
    4. **Correlation Bar Chart** - Direction and strength of relationships
    
    **Interactive Features:**
    - Move price slider → Updates histogram and scatter plot
    - Move age slider → Changes data points in all plots
    - Move correlation slider → Updates heatmap and bar chart colors
    
    *Visualizations created by: 23f2001189@ds.study.iitm.ac.in*
    """), fig


# ============================================================================
# CELL 10: Interactive Prediction Model (Depends on Cells 4, 5, 6, 7)
# ============================================================================
@app.cell
def _(filtered_data, ui, mo, np, pd):
    """
    ## Interactive Price Prediction Model
    
    This cell creates an interactive prediction tool that uses
    the filtered data and allows user input.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    
    # Prepare data for modeling
    if len(filtered_data) > 10:
        X = filtered_data[['square_footage', 'bedrooms', 'bathrooms', 'age', 'location_score']]
        y = filtered_data['price']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Create interactive input sliders
        sqft_slider = ui.slider(
            start=800, stop=4000, step=100, value=2000,
            label="📐 Square Footage", thumb_label=True
        )
        
        bed_slider = ui.slider(
            start=2, stop=5, step=1, value=3,
            label="🛏️ Bedrooms", thumb_label=True
        )
        
        bath_slider = ui.slider(
            start=1, stop=3, step=0.5, value=2,
            label="🚿 Bathrooms", thumb_label=True
        )
        
        pred_age_slider = ui.slider(
            start=0, stop=100, step=5, value=20,
            label="🏠 Property Age", thumb_label=True
        )
        
        loc_slider = ui.slider(
            start=0, stop=10, step=0.5, value=5,
            label="📍 Location Score", thumb_label=True
        )
        
        # Get current values
        input_values = np.array([
            sqft_slider.value,
            bed_slider.value,
            bath_slider.value,
            pred_age_slider.value,
            loc_slider.value
        ]).reshape(1, -1)
        
        # Scale and predict
        input_scaled = scaler.transform(input_values)
        predicted_price = model.predict(input_scaled)[0]
        
        # Calculate feature importance
        importance = pd.DataFrame({
            'Feature': ['Square Footage', 'Bedrooms', 'Bathrooms', 'Age', 'Location'],
            'Coefficient': model.coef_,
            'Impact': np.abs(model.coef_)
        }).sort_values('Impact', ascending=False)
        
        return mo.md(f"""
        ### 🎯 Interactive Price Predictor
        
        **Adjust these sliders to predict property prices:**
        
        {sqft_slider}
        
        {bed_slider}
        
        {bath_slider}
        
        {pred_age_slider}
        
        {loc_slider}
        
        **Prediction Result:**
        
        <div style="background: linear-gradient(135deg, #4CAF50, #8BC34A); 
                    padding: 25px; 
                    border-radius: 10px; 
                    color: white; 
                    text-align: center;
                    margin: 20px 0;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h2 style="color: white; margin: 0; font-size: 2em;">
            ${predicted_price:,.0f}
            </h2>
            <p style="margin: 10px 0 0 0; opacity: 0.9;">
            Estimated Property Value
            </p>
        </div>
        
        **Most Important Features:**
        1. **{importance.iloc[0]['Feature']}** (Impact: {abs(importance.iloc[0]['Coefficient']):.1f})
        2. **{importance.iloc[1]['Feature']}** (Impact: {abs(importance.iloc[1]['Coefficient']):.1f})
        3. **{importance.iloc[2]['Feature']}** (Impact: {abs(importance.iloc[2]['Coefficient']):.1f})
        
        **Model Details:**
        - Based on {len(filtered_data)} filtered properties
        - R² Score: {model.score(X_scaled, y):.3f}
        - All values update in real-time
        
        *Predictor created by: 23f2001189@ds.study.iitm.ac.in*
        """), model, predicted_price, importance, sqft_slider, bed_slider, bath_slider, pred_age_slider, loc_slider
    
    else:
        return mo.md("""
        ### ⚠️ Insufficient Data for Prediction
        
        Please adjust the filters to include more properties for accurate prediction.
        Currently only {len(filtered_data)} properties match your criteria.
        
        *Contact: 23f2001189@ds.study.iitm.ac.in*
        """), None, None, None, None, None, None, None, None


# ============================================================================
# CELL 11: Summary Dashboard (Depends on Multiple Cells)
# ============================================================================
@app.cell
def _(mo, count, percentage, avg_price, strongest_feature, strongest_value, 
      threshold, correlations):
    """
    ## Interactive Analysis Summary
    
    This cell provides a comprehensive summary that updates with all interactions.
    Shows dependencies on multiple previous cells.
    """
    return mo.md(f"""
    # 📊 Interactive Analysis Dashboard
    
    ## 📋 Current Analysis State
    
    **Data Summary:**
    - Properties analyzed: **{count}** ({percentage:.1f}% of total)
    - Average price: **${avg_price:,.0f}**
    
    **Analysis Parameters:**
    - Correlation threshold: **{threshold:.2f}**
    - Significant features: **{len([c for c in correlations.values() if abs(c) >= threshold])}** of {len(correlations)}
    
    ## 🎛️ Interactive Controls Used
    
    1. **Price Range Slider** - Filters properties by price
    2. **Age Range Slider** - Filters properties by age  
    3. **Correlation Threshold Slider** - Controls statistical sensitivity
    
    ## 📈 Key Findings
    
    **Strongest Relationship:**
    - Feature: **{strongest_feature.replace('_', ' ').title() if strongest_feature != 'N/A' else 'N/A'}**
    - Correlation: **{strongest_value:.3f}**
    - This explains price variation better than other features
    
    **Significant Correlations (≥ {threshold:.2f}):**
    {''.join([f"- {k.replace('_', ' ').title()}: {v:.3f}\\n" for k, v in correlations.items() if abs(v) >= threshold])}
    
    ## 🔄 Real-time Features
    
    This notebook demonstrates:
    - **Reactive programming** with Marimo
    - **Automatic updates** when sliders change
    - **Dynamic visualizations** that refresh in real-time
    - **Interactive statistical analysis**
    - **Live price prediction**
    
    ## 📧 Contact Information
    
    **Data Scientist:** 23f2001189@ds.study.iitm.ac.in
    
    **Notebook Features:**
    ✅ Multiple interactive slider widgets  
    ✅ Variable dependencies between cells  
    ✅ Dynamic markdown output  
    ✅ Real-time visualizations  
    ✅ Statistical modeling  
    ✅ Price prediction tool
    
    ---
    
    *Try adjusting any slider to see the entire analysis update automatically!*
    
    *Created: December 2023 | Version: 3.0*
    """)


# ============================================================================
# Run the application
# ============================================================================
if __name__ == "__main__":
    app.run()
