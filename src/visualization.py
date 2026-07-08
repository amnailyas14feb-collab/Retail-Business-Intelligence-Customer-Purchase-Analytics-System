import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple

def set_style():
    """Sets a clean, premium visual style for Matplotlib/Seaborn plots."""
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Inter', 'DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 150

def plot_heatmap(
    corr_matrix: pd.DataFrame, 
    cmap_theme: str = 'coolwarm', 
    annot: bool = True, 
    font_size: int = 10,
    title: str = "Correlation Heatmap"
) -> plt.Figure:
    """
    Generates a Pearson correlation heatmap with the upper triangle masked.
    
    Args:
        corr_matrix (pd.DataFrame): Calculated correlation matrix.
        cmap_theme (str): Seaborn colormap (e.g., 'coolwarm', 'RdBu_r', 'mako', 'vlag').
        annot (bool): Whether to annotate the cells with numbers.
        font_size (int): Size of the cell text.
        title (str): Heatmap title.
        
    Returns:
        plt.Figure: The Matplotlib Figure object.
    """
    set_style()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Mask the upper triangle (symmetric duplication)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Plot using Seaborn heatmap
    sns.heatmap(
        corr_matrix, 
        mask=mask, 
        cmap=cmap_theme, 
        vmax=1.0, 
        vmin=-1.0, 
        center=0,
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .8, "label": "Pearson Correlation Coefficient (r)"},
        annot=annot,
        fmt=".2f",
        annot_kws={"size": font_size, "weight": "bold"},
        ax=ax
    )
    
    ax.set_title(title, fontsize=16, pad=20, weight='bold', color='#1e293b')
    
    # Improve labels readability
    plt.xticks(rotation=45, ha='right', color='#475569')
    plt.yticks(rotation=0, color='#475569')
    
    fig.tight_layout()
    return fig


def plot_scatter(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    hue_col: Optional[str] = None, 
    show_regression: bool = True,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Creates a scatter plot with an optional regression line and categorical color grouping (hue).
    
    Args:
        df (pd.DataFrame): Data.
        x_col (str): X axis column name.
        y_col (str): Y axis column name.
        hue_col (Optional[str]): Column name for coloring scatter points.
        show_regression (bool): Whether to plot a linear regression trend line.
        title (Optional[str]): Custom plot title.
        
    Returns:
        plt.Figure: Matplotlib Figure object.
    """
    set_style()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Clean nulls in x and y for plotting
    req_cols = [x_col, y_col]
    if hue_col:
        req_cols.append(hue_col)
    plot_df = df[req_cols].dropna()
    
    # Plot using scatterplot (handles hue natively)
    if hue_col:
        sns.scatterplot(
            data=plot_df, 
            x=x_col, 
            y=y_col, 
            hue=hue_col, 
            palette="Set2",
            alpha=0.7, 
            edgecolor="w", 
            linewidth=0.5, 
            ax=ax
        )
        # Add regression line over the points if requested
        if show_regression:
            sns.regplot(
                data=plot_df, 
                x=x_col, 
                y=y_col, 
                ax=ax,
                scatter=False,
                color="#ef4444",
                line_kws={"linewidth": 2, "label": "Overall Trend Line"}
            )
            ax.legend()
    else:
        # Standard scatter with optional regression line
        if show_regression:
            sns.regplot(
                data=plot_df, 
                x=x_col, 
                y=y_col, 
                ax=ax,
                color="#3b82f6",
                scatter_kws={"alpha": 0.6, "edgecolor": "w", "linewidths": 0.5, "color": "#1e40af"},
                line_kws={"color": "#ef4444", "linewidth": 2, "label": "Trend Line"}
            )
            ax.legend()
        else:
            sns.scatterplot(
                data=plot_df, 
                x=x_col, 
                y=y_col, 
                color="#3b82f6",
                alpha=0.7, 
                edgecolor="w", 
                linewidth=0.5, 
                ax=ax
            )
            
    # Set titles & labels
    plt_title = title if title else f"Relationship: {x_col.replace('_', ' ')} vs {y_col.replace('_', ' ')}"
    ax.set_title(plt_title, fontsize=14, weight='bold', pad=15, color='#1e293b')
    ax.set_xlabel(x_col.replace('_', ' '), fontsize=11, color='#475569')
    ax.set_ylabel(y_col.replace('_', ' '), fontsize=11, color='#475569')
    
    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.5)
    
    fig.tight_layout()
    return fig


def plot_pairplot(
    df: pd.DataFrame, 
    columns: List[str], 
    hue_col: Optional[str] = None, 
    max_samples: int = 500
) -> plt.Figure:
    """
    Creates a scatter matrix (pair plot) for multiple columns.
    Downsamples the dataset for performance if it exceeds max_samples.
    
    Args:
        df (pd.DataFrame): Input dataset.
        columns (List[str]): Columns to include.
        hue_col (Optional[str]): Categorical column for point colors.
        max_samples (int): Max rows to plot (prevents lag).
        
    Returns:
        plt.Figure: Matplotlib Figure object.
    """
    set_style()
    
    # Filter list
    cols = [c for c in columns if c in df.columns]
    if hue_col and hue_col in df.columns:
        all_cols = cols + [hue_col]
    else:
        all_cols = cols
        
    plot_df = df[all_cols].dropna()
    
    # Downsample if needed
    if len(plot_df) > max_samples:
        plot_df = plot_df.sample(n=max_samples, random_state=42)
        
    # Generate pairplot
    g = sns.pairplot(
        plot_df, 
        vars=cols, 
        hue=hue_col, 
        palette="husl" if hue_col else None,
        diag_kind="kde", 
        plot_kws={"alpha": 0.6, "edgecolor": "w", "linewidth": 0.5}
    )
    
    # Adjust titles and spacing
    g.fig.suptitle(f"Scatter Matrix Analysis (n={len(plot_df)} samples)", y=1.02, fontsize=16, weight='bold', color='#1e293b')
    
    return g.fig


def save_plot_to_png(fig: plt.Figure, output_path: str, dpi: int = 300) -> bool:
    """
    Saves a Matplotlib/Seaborn figure to a PNG file.
    
    Args:
        fig (plt.Figure): Matplotlib figure object.
        output_path (str): File destination.
        dpi (int): Resolution.
        
    Returns:
        bool: True if save succeeded, False otherwise.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Avoid saving lmplot figures with multiple close calls or figure conflicts
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig) # free memory
        return True
    except Exception as e:
        print(f"Error saving plot to PNG: {str(e)}")
        return False
