"""
Week 6 - Interactive Sales Dashboard

This project demonstrates:
- Data loading and validation
- Data preparation
- Seaborn statistical visualizations
- Matplotlib subplot dashboard
- Plotly interactive visualizations
- KPI calculations
- Interactive dropdown filtering

Author: Viren
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ================================================================
# PROJECT CONFIGURATION
# ================================================================

DATA_FILE = Path("sales_data.csv")
OUTPUT_DIR = Path("visualizations")


# ================================================================
# DATA LOADING
# ================================================================

def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load the sales dataset from a CSV file.

    Parameters
    ----------
    file_path : Path
        Path to the CSV dataset.

    Returns
    -------
    pd.DataFrame
        Loaded sales dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset does not exist.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    return df


# ================================================================
# DATA VALIDATION
# ================================================================

def validate_data(df: pd.DataFrame) -> None:
    """
    Validate the basic structure and quality of the dataset.

    Checks:
    - Required columns
    - Missing values
    - Duplicate rows
    """

    required_columns = [
        "Date",
        "Product",
        "Quantity",
        "Price",
        "Customer_ID",
        "Region",
        "Total_Sales"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    missing_values = df.isnull().sum().sum()

    if missing_values > 0:
        print(
            f"Warning: Dataset contains "
            f"{missing_values} missing values."
        )

    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        print(
            f"Warning: Dataset contains "
            f"{duplicate_count} duplicate rows."
        )

    print("Data validation completed.")


# ================================================================
# DATA PREPARATION
# ================================================================

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataset for analysis and visualization.

    Converts Date to datetime and ensures numerical
    columns contain numeric values.

    Parameters
    ----------
    df : pd.DataFrame
        Raw sales dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """

    df = df.copy()

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # Convert numerical columns
    numeric_columns = [
        "Quantity",
        "Price",
        "Total_Sales"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove rows where essential values are invalid
    df = df.dropna(
        subset=[
            "Date",
            "Quantity",
            "Price",
            "Total_Sales"
        ]
    )

    return df


# ================================================================
# VISUALIZATION CONFIGURATION
# ================================================================

def configure_visualization_style() -> None:
    """
    Configure the global Seaborn visualization theme.
    """

    sns.set_theme(
        style="whitegrid",
        context="notebook"
    )


def create_output_directory() -> None:
    """
    Create the visualization output directory.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ================================================================
# KPI CALCULATIONS
# ================================================================

def calculate_kpis(df: pd.DataFrame) -> dict:
    """
    Calculate key performance indicators.

    Returns
    -------
    dict
        Dashboard KPI values.
    """

    kpis = {
        "total_sales": df["Total_Sales"].sum(),
        "total_quantity": df["Quantity"].sum(),
        "average_sale": df["Total_Sales"].mean(),
        "total_customers": df["Customer_ID"].nunique()
    }

    return kpis


# ================================================================
# SEABORN CHART 1
# ================================================================

def create_sales_by_product_chart(
    df: pd.DataFrame
) -> None:
    """
    Create a bar chart showing total sales by product.
    """

    product_sales = (
        df.groupby("Product")["Total_Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x=product_sales.index,
        y=product_sales.values
    )

    plt.title(
        "Total Sales by Product",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel("Product")
    plt.ylabel("Total Sales")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "sales_by_product.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ================================================================
# SEABORN CHART 2
# ================================================================

def create_sales_trend_chart(
    df: pd.DataFrame
) -> None:
    """
    Create a line chart showing daily sales trends.
    """

    daily_sales = (
        df.groupby("Date")["Total_Sales"]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(12, 6))

    sns.lineplot(
        data=daily_sales,
        x="Date",
        y="Total_Sales",
        marker="o"
    )

    plt.title(
        "Daily Sales Trend",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel("Date")
    plt.ylabel("Total Sales")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "sales_trend.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ================================================================
# SEABORN CHART 3
# ================================================================

def create_box_plot(
    df: pd.DataFrame
) -> None:
    """
    Create a box plot showing price distribution
    across products.
    """

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="Product",
        y="Price"
    )

    plt.title(
        "Price Distribution by Product",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel("Product")
    plt.ylabel("Price")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "price_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ================================================================
# SEABORN CHART 4
# ================================================================

def create_violin_plot(
    df: pd.DataFrame
) -> None:
    """
    Create a violin plot showing sales distribution
    across regions.
    """

    plt.figure(figsize=(10, 6))

    sns.violinplot(
        data=df,
        x="Region",
        y="Total_Sales"
    )

    plt.title(
        "Sales Distribution by Region",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel("Region")
    plt.ylabel("Total Sales")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "sales_violin.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ================================================================
# SEABORN CHART 5
# ================================================================

def create_correlation_heatmap(
    df: pd.DataFrame
) -> None:
    """
    Create a correlation heatmap for numerical variables.
    """

    numeric_data = df[
        [
            "Quantity",
            "Price",
            "Total_Sales"
        ]
    ]

    correlation_matrix = numeric_data.corr()

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        linewidths=0.5
    )

    plt.title(
        "Sales Data Correlation Heatmap",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ================================================================
# 2 × 2 MULTI-PLOT DASHBOARD
# ================================================================

def create_dashboard(
    df: pd.DataFrame
) -> None:
    """
    Create a 2 × 2 dashboard containing:
    - Product sales
    - Regional sales
    - Price distribution
    - Correlation heatmap
    """

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 12)
    )

    # ------------------------------------------------------------
    # Plot 1: Sales by Product
    # ------------------------------------------------------------

    product_sales = (
        df.groupby("Product")["Total_Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    sns.barplot(
        x=product_sales.index,
        y=product_sales.values,
        ax=axes[0, 0]
    )

    axes[0, 0].set_title(
        "Sales by Product",
        fontweight="bold"
    )

    axes[0, 0].set_xlabel("Product")
    axes[0, 0].set_ylabel("Total Sales")

    axes[0, 0].tick_params(
        axis="x",
        rotation=20
    )

    # ------------------------------------------------------------
    # Plot 2: Sales by Region
    # ------------------------------------------------------------

    region_sales = (
        df.groupby("Region")["Total_Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    sns.barplot(
        x=region_sales.index,
        y=region_sales.values,
        ax=axes[0, 1]
    )

    axes[0, 1].set_title(
        "Sales by Region",
        fontweight="bold"
    )

    axes[0, 1].set_xlabel("Region")
    axes[0, 1].set_ylabel("Total Sales")

    # ------------------------------------------------------------
    # Plot 3: Price Distribution
    # ------------------------------------------------------------

    sns.boxplot(
        data=df,
        x="Product",
        y="Price",
        ax=axes[1, 0]
    )

    axes[1, 0].set_title(
        "Price Distribution",
        fontweight="bold"
    )

    axes[1, 0].set_xlabel("Product")
    axes[1, 0].set_ylabel("Price")

    axes[1, 0].tick_params(
        axis="x",
        rotation=20
    )

    # ------------------------------------------------------------
    # Plot 4: Correlation Heatmap
    # ------------------------------------------------------------

    numeric_data = df[
        [
            "Quantity",
            "Price",
            "Total_Sales"
        ]
    ]

    correlation_matrix = numeric_data.corr()

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=axes[1, 1]
    )

    axes[1, 1].set_title(
        "Correlation Heatmap",
        fontweight="bold"
    )

    # ------------------------------------------------------------
    # Final dashboard formatting
    # ------------------------------------------------------------

    fig.suptitle(
        "Sales Dashboard - Statistical Overview",
        fontsize=20,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "dashboard.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ================================================================
# PLOTLY INTERACTIVE CHART 1
# ================================================================

def create_interactive_product_chart(
    df: pd.DataFrame
) -> None:
    """
    Create an interactive bar chart for product sales.
    """

    product_sales = (
        df.groupby(
            "Product",
            as_index=False
        )["Total_Sales"]
        .sum()
    )

    fig = px.bar(
        product_sales,
        x="Product",
        y="Total_Sales",
        title="Interactive Sales by Product",
        hover_data=["Total_Sales"]
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Product",
        yaxis_title="Total Sales"
    )

    fig.write_html(
        OUTPUT_DIR / "interactive_product_sales.html"
    )


# ================================================================
# PLOTLY INTERACTIVE CHART 2
# ================================================================

def create_interactive_sales_trend(
    df: pd.DataFrame
) -> None:
    """
    Create an interactive sales trend chart.
    """

    fig = px.line(
        df,
        x="Date",
        y="Total_Sales",
        color="Region",
        markers=True,
        hover_data=[
            "Product",
            "Quantity",
            "Price",
            "Customer_ID"
        ],
        title="Interactive Sales Trend by Region"
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Total Sales"
    )

    fig.write_html(
        OUTPUT_DIR / "interactive_sales_trend.html"
    )


# ================================================================
# PLOTLY INTERACTIVE CHART 3
# ================================================================

def create_interactive_region_chart(
    df: pd.DataFrame
) -> None:
    """
    Create an interactive donut chart for regional sales.
    """

    region_sales = (
        df.groupby(
            "Region",
            as_index=False
        )["Total_Sales"]
        .sum()
    )

    fig = px.pie(
        region_sales,
        names="Region",
        values="Total_Sales",
        title="Sales Distribution by Region",
        hole=0.4
    )

    fig.update_layout(
        template="plotly_white"
    )

    fig.write_html(
        OUTPUT_DIR / "interactive_region_sales.html"
    )


# ================================================================
# COMPLETE INTERACTIVE DASHBOARD
# ================================================================

def create_interactive_dashboard(
    df: pd.DataFrame
) -> None:
    """
    Create the complete interactive Plotly dashboard.

    Includes:
    - KPI cards
    - Product sales
    - Regional sales
    - Daily sales trend
    - Product dropdown filter
    - Hover information
    """

    # ------------------------------------------------------------
    # KPI calculations
    # ------------------------------------------------------------

    kpis = calculate_kpis(df)

    total_sales = kpis["total_sales"]
    total_quantity = kpis["total_quantity"]
    average_sale = kpis["average_sale"]
    total_customers = kpis["total_customers"]

    # ------------------------------------------------------------
    # Aggregated data
    # ------------------------------------------------------------

    product_sales = (
        df.groupby(
            "Product",
            as_index=False
        )["Total_Sales"]
        .sum()
        .sort_values(
            "Total_Sales",
            ascending=False
        )
    )

    region_sales = (
        df.groupby(
            "Region",
            as_index=False
        )["Total_Sales"]
        .sum()
        .sort_values(
            "Total_Sales",
            ascending=False
        )
    )

    daily_sales = (
        df.groupby(
            "Date",
            as_index=False
        )["Total_Sales"]
        .sum()
    )

    # ------------------------------------------------------------
    # Create dashboard grid
    # ------------------------------------------------------------

    fig = make_subplots(
        rows=4,
        cols=2,
        specs=[
            [
                {"type": "indicator"},
                {"type": "indicator"}
            ],
            [
                {"type": "indicator"},
                {"type": "indicator"}
            ],
            [
                {"type": "bar"},
                {"type": "pie"}
            ],
            [
                {"type": "scatter", "colspan": 2},
                None
            ]
        ],
        subplot_titles=[
            "Total Sales",
            "Units Sold",
            "Average Sale",
            "Unique Customers",
            "Sales by Product",
            "Sales by Region",
            "Daily Sales Trend"
        ],
        vertical_spacing=0.08
    )

    # ------------------------------------------------------------
    # KPI 1
    # ------------------------------------------------------------

    fig.add_trace(
        go.Indicator(
            mode="number",
            value=total_sales,
            number={
                "prefix": "₹",
                "valueformat": ",.0f"
            },
            title={
                "text": "Total Sales"
            }
        ),
        row=1,
        col=1
    )

    # ------------------------------------------------------------
    # KPI 2
    # ------------------------------------------------------------

    fig.add_trace(
        go.Indicator(
            mode="number",
            value=total_quantity,
            number={
                "valueformat": ",.0f"
            },
            title={
                "text": "Units Sold"
            }
        ),
        row=1,
        col=2
    )

    # ------------------------------------------------------------
    # KPI 3
    # ------------------------------------------------------------

    fig.add_trace(
        go.Indicator(
            mode="number",
            value=average_sale,
            number={
                "prefix": "₹",
                "valueformat": ",.0f"
            },
            title={
                "text": "Average Sale"
            }
        ),
        row=2,
        col=1
    )

    # ------------------------------------------------------------
    # KPI 4
    # ------------------------------------------------------------

    fig.add_trace(
        go.Indicator(
            mode="number",
            value=total_customers,
            number={
                "valueformat": ",.0f"
            },
            title={
                "text": "Unique Customers"
            }
        ),
        row=2,
        col=2
    )

    # ------------------------------------------------------------
    # Product Sales
    # ------------------------------------------------------------

    fig.add_trace(
        go.Bar(
            x=product_sales["Product"],
            y=product_sales["Total_Sales"],
            name="Product Sales",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Sales: ₹%{y:,.0f}"
                "<extra></extra>"
            )
        ),
        row=3,
        col=1
    )

    # ------------------------------------------------------------
    # Regional Sales
    # ------------------------------------------------------------

    fig.add_trace(
        go.Pie(
            labels=region_sales["Region"],
            values=region_sales["Total_Sales"],
            hole=0.4,
            name="Regional Sales",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Sales: ₹%{value:,.0f}<br>"
                "Share: %{percent}"
                "<extra></extra>"
            )
        ),
        row=3,
        col=2
    )

    # ------------------------------------------------------------
    # Daily Sales Trend
    # ------------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=daily_sales["Date"],
            y=daily_sales["Total_Sales"],
            mode="lines+markers",
            name="Daily Sales",
            hovertemplate=(
                "<b>%{x|%Y-%m-%d}</b><br>"
                "Sales: ₹%{y:,.0f}"
                "<extra></extra>"
            )
        ),
        row=4,
        col=1
    )

    # ------------------------------------------------------------
    # Product dropdown
    # ------------------------------------------------------------

    dropdown_buttons = [
        dict(
            label="All Products",
            method="update",
            args=[
                {
                    "x": [
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        daily_sales["Date"]
                    ],
                    "y": [
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        daily_sales["Total_Sales"]
                    ]
                }
            ]
        )
    ]

    for product in sorted(df["Product"].unique()):

        filtered_sales = (
            df[df["Product"] == product]
            .groupby(
                "Date",
                as_index=False
            )["Total_Sales"]
            .sum()
        )

        dropdown_buttons.append(
            dict(
                label=product,
                method="update",
                args=[
                    {
                        "x": [
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            filtered_sales["Date"]
                        ],
                        "y": [
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            filtered_sales["Total_Sales"]
                        ]
                    }
                ]
            )
        )

    # ------------------------------------------------------------
    # Dashboard layout
    # ------------------------------------------------------------

    fig.update_layout(
        title={
            "text": "Interactive Sales Dashboard",
            "x": 0.5,
            "xanchor": "center",
            "font": {
                "size": 28
            }
        },
        height=1250,
        width=1400,
        template="plotly_white",
        showlegend=False,
        margin={
            "l": 60,
            "r": 60,
            "t": 120,
            "b": 60
        },
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=0.02,
                y=1.06,
                xanchor="left",
                yanchor="top",
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1
            )
        ]
    )

    # ------------------------------------------------------------
    # Axis labels
    # ------------------------------------------------------------

    fig.update_xaxes(
        title_text="Product",
        row=3,
        col=1
    )

    fig.update_yaxes(
        title_text="Sales (₹)",
        row=3,
        col=1
    )

    fig.update_xaxes(
        title_text="Date",
        row=4,
        col=1
    )

    fig.update_yaxes(
        title_text="Sales (₹)",
        row=4,
        col=1
    )

    # ------------------------------------------------------------
    # Save dashboard
    # ------------------------------------------------------------

    output_file = (
        OUTPUT_DIR /
        "interactive_dashboard.html"
    )

    fig.write_html(
        output_file,
        include_plotlyjs=True
    )

    print(
        f"Interactive dashboard saved to: "
        f"{output_file}"
    )


# ================================================================
# MAIN PROGRAM
# ================================================================

def main() -> None:
    """
    Execute the complete sales dashboard workflow.
    """

    print("=" * 60)
    print("WEEK 6 - INTERACTIVE SALES DASHBOARD")
    print("=" * 60)

    # ------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------

    print("\n[1/6] Loading dataset...")

    df = load_data(DATA_FILE)

    print(
        f"Dataset loaded successfully: "
        f"{df.shape[0]} rows × {df.shape[1]} columns"
    )

    # ------------------------------------------------------------
    # Step 2: Validate data
    # ------------------------------------------------------------

    print("\n[2/6] Validating dataset...")

    validate_data(df)

    # ------------------------------------------------------------
    # Step 3: Prepare data
    # ------------------------------------------------------------

    print("\n[3/6] Preparing data...")

    df = prepare_data(df)

    print("Data preparation completed.")

    # ------------------------------------------------------------
    # Step 4: Configure visualization environment
    # ------------------------------------------------------------

    print("\n[4/6] Configuring visualization environment...")

    configure_visualization_style()
    create_output_directory()

    print("Visualization environment ready.")

    # ------------------------------------------------------------
    # Step 5: Create Seaborn visualizations
    # ------------------------------------------------------------

    print("\n[5/6] Creating statistical visualizations...")

    create_sales_by_product_chart(df)

    print("✓ Sales by product chart")

    create_sales_trend_chart(df)

    print("✓ Sales trend chart")

    create_box_plot(df)

    print("✓ Box plot")

    create_violin_plot(df)

    print("✓ Violin plot")

    create_correlation_heatmap(df)

    print("✓ Correlation heatmap")

    create_dashboard(df)

    print("✓ 2 × 2 dashboard")

    # ------------------------------------------------------------
    # Step 6: Create Plotly visualizations
    # ------------------------------------------------------------

    print("\n[6/6] Creating interactive visualizations...")

    create_interactive_product_chart(df)

    print("✓ Interactive product chart")

    create_interactive_sales_trend(df)

    print("✓ Interactive sales trend")

    create_interactive_region_chart(df)

    print("✓ Interactive region chart")

    create_interactive_dashboard(df)

    print("✓ Complete interactive dashboard")

    # ------------------------------------------------------------
    # Final message
    # ------------------------------------------------------------

    print("\n" + "=" * 60)
    print("DASHBOARD CREATION COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"\nAll visualization files are available in:\n"
        f"{OUTPUT_DIR.resolve()}"
    )


# ================================================================
# PROGRAM ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()