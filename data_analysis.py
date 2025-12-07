"""
# Interactive Data Analysis Notebook
## Research Institution - Data Science Division

**Analyst:** 23f2001189@ds.study.iitm.ac.in
**Dataset:** Housing Price Analysis
**Created:** December 2023
"""

import marimo

__generated_with = "0.1.64"
app = marimo.App()


# ============================================================================
# CELL 1: Import Libraries and Setup
# ============================================================================
@app.cell
def _():
    """
    Import libraries and setup environment.
    Analyst: 23f2001189@ds.study.iitm.ac.in
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Import marimo UI components
    import marimo as mo
    
    print("✅ Libraries imported successfully")
    return mo, np, pd, plt, sns


# ============================================================================
# CELL 2: Create Dataset
# ============================================================================
@app.cell
def _(np, pd):
    """
    Create synthetic housing dataset.
    Analyst: 23f2001189@ds.study.iitm.ac.in
    """
    np.random.seed(42)
    n_samples = 200
    
    # Generate features
    size = np.random.normal(2000, 500, n_samples).clip(800, 4000)
    rooms = np.random.choice([2, 3, 4], n_samples, p=[0.2, 0.6, 0.2])
    age = np.random.exponential(15, n_samples).clip(0, 50)
    location = np.random.uniform(0, 10, n_samples)
    
    # Generate price with relationships
    price = (
        100000
        + 200 * size
        + 30000 * rooms
        - 1500 * age
        + 20000 * location
        + np.random.normal(0, 30000, n_samples)
    ).clip(150000, 800000)
    
    data = pd.DataFrame({
        'price': price,
        'size': size,
        'rooms': rooms,
        'age': age,
        'location': location
    })
    
    return data


# ============================================================================
# CELL 3: INTERACTIVE SLIDER WIDGET - PRICE THRESHOLD
# ============================================================================
@app.cell
def _(mo):
    """
    INTERACTIVE SLIDER WIDGET for price threshold filtering.
    This widget allows real-time interaction with the data.
    Analyst: 23f2001189@ds.study.iitm.ac.in
    """
    # Create an interactive slider widget
    price_threshold_slider = mo.ui.slider(
        start=150000,      # Minimum value
        stop=800000,       # Maximum value
        value=400000,      # Default value
        step=50000,        # Increment step
        label="💰 Price Threshold Filter",
        show_value=True    # Display current value
    )
    
    # Display the slider
    return price_threshold_slider


# ============================================================================
# CELL 4: Filter Data Based on Slider (Depends on Cells 2 and 3)
# ============================================================================
@app.cell
def _(data, price_threshold_slider, mo):
    """
    Filter data based on interactive slider value.
    This cell shows variable dependency between cells.
    Analyst: 23f2001189@ds.study.iitm.ac.in
    """
    # Get current slider value
    threshold = price_threshold_slider.value
    
    # Filter data
    filtered_data = data[data['price'] >= threshold]
    
    # Calculate statistics
    count = len(filtered_data)
    avg_price = filtered_data['price'].mean() if count > 0 else 0
    percentage = (count / len(data) * 100) if len(data) > 0 else 0
    
    # Dynamic markdown output based on slider state
    return mo.md(f"""
    ### 📊 Filtered Data Analysis
    
    **Current Price Threshold:** ${threshold:,}
    
    **Results:**
    - Properties above threshold: **{count}**
    - Percentage of total: **{percentage:.1f}%**
    - Average price: **${avg_price:,.0f}**
    
    **Price Statistics:**
    - Minimum: ${filtered_data['price'].min():,.0f} if count > 0 else "N/A"
    - Maximum: ${filtered_data['price'].max():,.0f} if count > 0 else "N/A"
    - Median: ${filtered_data['price'].median():,.0f} if count > 0 else "N/A"
    
    **Try moving the slider above to see real-time updates!**
    
    *Analysis by: 23f2001189@ds.study.iitm.ac.in*
    """), filtered_data, threshold, count, avg_price, percentage


# ============================================================================
# CELL 5: INTERACTIVE SLIDER WIDGET - CORRELATION THRESHOLD
# ============================================================================
@app.cell
def _(mo):
    """
    SECOND INTERACTIVE SLIDER WIDGET for correlation analysis.
    Analyst: 23f2001189@ds.study.iitm.ac.in
    """
    # Create second interactive slider
    correlation_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        value=0.5,
        step=0.1,
        label="📊 Correlation Strength Threshold",
        show_value=True
    )
    
    return correlation_slider


# ============================================================================
# CELL 6: Correlation Analysis (Depends on Cells 4 and 5)
# ============================================================================
@app.cell
def _(filtered_data, correlation_slider, mo, np):
    """
    Perform correlation analysis with interactive threshold.
    Shows variable dependency between multiple cells.
    Analyst: 23f2001189@ds.study.iitm.ac.in
    """
    # Get correlation threshold
    corr_threshold = correlation_slider.value
    
    # Calculate correlations with price
    correlations = {}
    for column in filtered_data.columns:
        if column != 'price' and len(filtered_data) > 1:
            corr = np.corrcoef(filtered_data['price'], filtered_data[column])[0, 1]
            correlations[column] = corr
    
    # Find strong correlations
    strong_correlations = {k: v for k, v in correlations.items() if abs(v) >= corr_threshold}
    
    # Dynamic output based on slider
    return mo.md(f"""
    ### 📈 Correlation Analysis
    
    **Current Threshold:** {corr_threshold}
    
    **Correlations with Price:**
    {''.join([f"- **{col}**: {corr:.3f} {'✅' if abs(corr) >= corr_threshold else '⚠️'}\\n" for col, corr in correlations.items()])}
    
    **Strong Correlations (≥ {corr_threshold}):**
    {''.join([f"- {col}: {corr:.3f}\\n" for col, corr in strong_correlations.items()]) if strong_correlations else "None"}
    
    **Interpretation:**
    - Values close to 1: Strong positive relationship
    - Values close to -1: Strong negative relationship
    - Values near 0: Weak relationship
    
    *Adjust the correlation slider above to filter relationships!*
    
    *Contact: 23f2001189@ds.study.iitm.ac.in*
    """), correlations, strong_correlations, corr_threshold


# ============================================================================
# CELL 7: Visualization (Depends on Cells 4, 5, 6)
# ============================================================================
@app.cell
def _(filtered_data, plt, sns, correlations, corr_threshold, mo):
    """
    Create interactive visualizations.
    Shows complex dependencies between multiple cells.
    Analyst: 23f2001189@ds.study.iitm.ac.in
    """
    if len(filtered_data) > 0:
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Price distribution
        axes[0].hist(filtered_data['price'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_xlabel('Price ($)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Price Distribution\n(Threshold: ${filtered_data["price"].min():,.0f}+)')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Correlation bar chart
        if correlations:
            features = list(correlations.keys())
            values = list(correlations.values())
            
            colors = ['green' if abs(v) >= corr_threshold else 'gray' for v in values]
            bars = axes[1].barh(features, values, color=colors, alpha=0.7)
            
            axes[1].axvline(x=0, color='black', linewidth=0.8)
            axes[1].axvline(x=corr_threshold, color='red', linestyle='--', alpha=0.5)
            axes[1].axvline(x=-corr_threshold, color='red', linestyle='--', alpha=0.5)
            
            axes[1].set_xlabel('Correlation with Price')
            axes[1].set_title(f'Feature Correlations\n(Threshold: {corr_threshold})')
            axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        return mo.md(f"""
        ### 📊 Interactive Visualizations
        
        **Plots update automatically when you adjust the sliders!**
        
        **Left:** Price distribution of filtered properties
        **Right:** Correlation of each feature with price (green = above threshold)
        
        *Move the sliders to see real-time updates!*
        
        *Visualizations by: 23f2001189@ds.study.iitm.ac.in*
        """), fig
    else:
        return mo.md("""
        ### ⚠️ No Data to Visualize
        
        Please adjust the price threshold slider to include more properties.
        
        *Contact: 23f2001189@ds.study.iitm.ac.in*
        """), None


# ============================================================================
# CELL 8: Interactive Prediction (Depends on Cells 2, 4)
# ============================================================================
@app.cell
def _(mo, filtered_data):
    """
    Interactive prediction based on filtered data.
    Analyst: 23f2001189@ds.study.iitm.ac.in
    """
    from sklearn.linear_model import LinearRegression
    
    if len(filtered_data) > 10:
        # Prepare data
        X = filtered_data[['size', 'rooms', 'age', 'location']]
        y = filtered_data['price']
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Create prediction sliders
        size_slider = mo.ui.slider(
            start=800, stop=4000, value=2000, step=100,
            label="🏠 Size (sq ft)"
        )
        
        rooms_slider = mo.ui.slider(
            start=2, stop=4, value=3, step=1,
            label="🛏️ Rooms"
        )
        
        age_slider = mo.ui.slider(
            start=0, stop=50, value=15, step=5,
            label="📅 Age (years)"
        )
        
        location_slider = mo.ui.slider(
            start=0, stop=10, value=5, step=0.5,
            label="📍 Location Score"
        )
        
        # Calculate prediction
        input_data = [[
            size_slider.value,
            rooms_slider.value,
            age_slider.value,
            location_slider.value
        ]]
        
        predicted_price = model.predict(input_data)[0]
        
        return mo.md(f"""
        ### 🎯 Price Prediction
        
        **Adjust these values to predict property price:**
        
        {size_slider}
        
        {rooms_slider}
        
        {age_slider}
        
        {location_slider}
        
        **Predicted Price:** **${predicted_price:,.0f}**
        
        **Based on:**
        - Model trained on {len(filtered_data)} properties
        - R² Score: {model.score(X, y):.3f}
        
        **Feature Importance:**
        - Size coefficient: ${model.coef_[0]:.0f} per sq ft
        - Rooms coefficient: ${model.coef_[1]:.0f} per room
        - Age coefficient: ${model.coef_[2]:.0f} per year
        - Location coefficient: ${model.coef_[3]:.0f} per point
        
        *Try adjusting the values above for different predictions!*
        
        *Predictor by: 23f2001189@ds.study.iitm.ac.in*
        """), model, predicted_price, size_slider, rooms_slider, age_slider, location_slider
    
    return mo.md("""
    ### ⚠️ Need More Data
    
    Please lower the price threshold to include more properties for prediction.
    
    *Contact: 23f2001189@ds.study.iitm.ac.in*
    """), None, None, None, None, None, None


# ============================================================================
# CELL 9: Summary Dashboard (Depends on Multiple Cells)
# ============================================================================
@app.cell
def _(mo, count, percentage, avg_price, correlations, corr_threshold):
    """
    Summary dashboard showing dependencies between cells.
    Analyst: 23f2001189@ds.study.iitm.ac.in
    """
    # Count strong correlations
    strong_count = len([c for c in correlations.values() if abs(c) >= corr_threshold]) if correlations else 0
    
    return mo.md(f"""
    # 📊 Analysis Summary
    
    ## Current Analysis State
    
    **Data Summary:**
    - Properties analyzed: **{count}**
    - Percentage of total: **{percentage:.1f}%**
    - Average price: **${avg_price:,.0f}**
    
    **Analysis Parameters:**
    - Correlation threshold: **{corr_threshold}**
    - Strong correlations found: **{strong_count}**
    
    ## Interactive Features
    
    ✅ **Interactive Slider Widgets:**
    1. Price Threshold Filter - Controls data filtering
    2. Correlation Threshold - Controls statistical analysis
    3. Prediction Inputs - Four sliders for price prediction
    
    ✅ **Variable Dependencies:**
    - Cell 4 depends on Cells 2 and 3
    - Cell 6 depends on Cells 4 and 5  
    - Cell 7 depends on Cells 4, 5, and 6
    - Cell 8 depends on Cell 4
    - This cell depends on Cells 4 and 6
    
    ✅ **Dynamic Output:**
    - All markdown updates in real-time
    - Visualizations refresh automatically
    - Statistics recalculate on slider changes
    
    ## Key Insights
    
    The analysis shows how different features relate to housing prices.
    Adjust the sliders to explore different aspects of the data.
    
    ## Contact Information
    
    **Data Scientist:** 23f2001189@ds.study.iitm.ac.in
    
    **Notebook Features:**
    - Interactive slider widgets
    - Variable dependencies between cells
    - Dynamic markdown output
    - Real-time visualizations
    - Statistical analysis
    - Price prediction
    
    ---
    
    *This notebook demonstrates reactive programming with Marimo.
    All components update automatically when sliders are adjusted.*
    
    *Created: December 2023*
    """)


# ============================================================================
# Run the application
# ============================================================================
if __name__ == "__main__":
    app.run()
