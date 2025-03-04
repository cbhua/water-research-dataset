import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from tqdm import tqdm
from scipy import stats
from typing import Union
from statsmodels.tsa.stattools import acf
from matplotlib.colors import LinearSegmentedColormap


def calculate_correlation_matrix(
        data: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
    """
    Calculate correlation matrix between all features for numpy array or pandas DataFrame.
    
    Args:
        data <pd.DataFrame/np.array>: Input data with shape [length, num_feats]
        
    Returns:
        <np.ndarray>: Correlation matrix with shape [num_feats, num_feats]
    """
    if isinstance(data, pd.DataFrame):
        correlation_matrix = data.corr()
    elif isinstance(data, np.ndarray):
        # Standardize the features
        standardized_data = (data - data.mean()) / data.std()
        # Calculate correlation matrix
        num_features = data.shape[1]
        correlation_matrix = np.zeros((num_features, num_features))
        for i in range(num_features):
            for j in range(num_features):
                # Correlation = mean of the product of standardized variables
                correlation_matrix[i, j] = np.mean(
                    standardized_data.iloc[:, i] * standardized_data.iloc[:, j]
                )
    
    return correlation_matrix.to_numpy()  


def create_custom_colormap():
    """Create a custom blue-white-red colormap"""
    colors = ['#3B4992', '#FFFFFF', '#EE0000']  # Blue to white to red
    nodes = [0.0, 0.5, 1.0]
    return LinearSegmentedColormap.from_list("custom", list(zip(nodes, colors)))


def plot_correlation_matrix(
        corr_matrix: np.ndarray, 
        feat_name: list,
        save_path: str
    ):
    """
    Args:
        - corr_matrix <np.ndarray> [num_feats, num_feats]: Correlation matrix with shape 
        - feat_name <list> [num_feats]: List of feature names
        - save_path <str>: Path to save the plot figure
    """
    # Mask upper triangle
    mask = np.triu(np.ones_like(corr_matrix), k=1)
    corr_matrix[mask == 1] = np.nan  # Set upper triangle to NaN

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create custom colormap
    cmap = create_custom_colormap()

    # Plot heatmap
    im = ax.imshow(corr_matrix, cmap=cmap, aspect='equal', vmin=-1, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)

    # Add correlation values
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            if not np.isnan(corr_matrix[i, j]):  # Only show lower triangle
                # Choose text color based on background
                color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                ha='center', va='center', color=color)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(feat_name)))
    ax.set_yticks(np.arange(len(feat_name)))
    ax.set_xticklabels(feat_name, rotation=90, ha='right')
    ax.set_yticklabels(feat_name)

    for spine in ax.spines.values():
        spine.set_color('white')

    ax.tick_params(color='white', which='both')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)


def calculate_lag_correlations(data, max_lag=5):
    """
    Calculate correlations between the time series and its lagged versions.
    
    Parameters:
    -----------
    data : array-like
        1D time series data
    max_lag : int
        Maximum number of lags to calculate
        
    Returns:
    --------
    dict
        Dictionary containing lag correlations
    """
    correlations = {}
    for lag in range(1, max_lag + 1):
        if len(data) > lag:
            correlation = np.corrcoef(data[lag:], data[:-lag])[0, 1]
            correlations[f'lag_{lag}_correlation'] = correlation
    return correlations


def calculate_1d_statistic(data):
    """
    Calculate comprehensive statistics for a 1D time series array.

    Args:
        - data <np.array>: 1D time series data

    Returns:
        - all_stats <dict>: Dictionary containing various statistical measures
    """
    # Convert to numpy array if not already
    ts = np.array(data)
    
    # Calculate lag correlations
    lag_correlations = calculate_lag_correlations(ts)
    
    # Basic statistics
    basic_stats = {
        'mean': np.mean(ts),
        'std': np.std(ts),
        'variance': np.var(ts),
        'min': np.min(ts),
        'max': np.max(ts),
        'range': np.ptp(ts),  # Peak to peak (max - min)
        'median': np.median(ts),
        'q25': np.percentile(ts, 25),  # 25th percentile
        'q75': np.percentile(ts, 75),  # 75th percentile
        'iqr': stats.iqr(ts),  # Interquartile range
    }
    
    # Add lag correlations to basic stats
    basic_stats.update(lag_correlations)
    
    # Add serial correlation (correlation between consecutive points)
    if len(ts) > 1:
        basic_stats['serial_correlation'] = np.corrcoef(ts[1:], ts[:-1])[0, 1]
    else:
        basic_stats['serial_correlation'] = None
    
    # Distribution statistics
    distribution_stats = {
        'skewness': stats.skew(ts),  # Measure of asymmetry
        'kurtosis': stats.kurtosis(ts),  # Measure of heavy-tailedness
        'is_normal': stats.normaltest(ts)[1],  # p-value for normality test
    }
    
    # Time series specific statistics
    # Calculate autocorrelation for different lags
    max_lag = min(len(ts) - 1, 10)  # Use up to 10 lags or series length - 1
    try:
        autocorr = acf(ts, nlags=max_lag, fft=True)  # Added fft=True for faster computation
        ts_specific_stats = {
            'autocorr_lag1': autocorr[1] if len(autocorr) > 1 else None,  # First-order autocorrelation
            'autocorr_lag2': autocorr[2] if len(autocorr) > 2 else None,
            'autocorr_lag3': autocorr[3] if len(autocorr) > 3 else None,
            'trend': np.polyfit(np.arange(len(ts)), ts, 1)[0],  # Linear trend coefficient
        }
    except:
        # Fallback if autocorrelation calculation fails
        ts_specific_stats = {
            'autocorr_lag1': None,
            'autocorr_lag2': None,
            'autocorr_lag3': None,
            'trend': np.polyfit(np.arange(len(ts)), ts, 1)[0],  # Linear trend coefficient
        }
    
    # Stationarity measures
    rolling_mean = np.mean(ts[:len(ts)//2]) - np.mean(ts[len(ts)//2:])
    rolling_std = np.std(ts[:len(ts)//2]) - np.std(ts[len(ts)//2:])
    
    stationarity_stats = {
        'rolling_mean_diff': rolling_mean,  # Difference in mean between first and second half
        'rolling_std_diff': rolling_std,    # Difference in std between first and second half
    }
    
    # Combine all statistics all together into one dict
    all_stats = {**basic_stats, **distribution_stats, **ts_specific_stats, **stationarity_stats}
    
    return all_stats


def calculate_all_statistic(data: pd.DataFrame):
    """
    Calculate all features data statistics.

    Args:
        - data <pd.DataFrame>: Input data with shape [length, num_feats]

    Returns:
        - all_stats <dict>: Dictionary containing all statistical measures

    Note:
        - Don't include the time column.
    """
    # Initialize dictionary to store statistics
    all_stats = {}
    for col in tqdm(data.columns):
        all_stats[col] = calculate_1d_statistic(data[col].values)
    
    return all_stats


def convert_statistic_to_csv(all_stats, save_path: str):
    """
    Convert the statistics dictionary to a pandas DataFrame and save to CSV.
    
    Args:
        - all_stats <dict>: Dictionary containing all statistical measures
    """
    # Create a DataFrame from the dictionary
    df = pd.DataFrame(all_stats).T

    # Add the first column name as "feature"
    df.index.name = 'feature'

    # Save to CSV
    df.to_csv(save_path)


def plot_1d(date, value, feat_name, save_path=None):
    """
    Plot a 1D time series data with date on x-axis and value on y-axis.

    Args:
        - date <np.array>: Date values
        - value <np.array>: Time series values
        - feat_name <str>: Feature name
        - save_path <str>: Path to save the plot figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # Use latex setting
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    _, ax = plt.subplots(1, figsize=(15, 3))

    ax.plot(date, value, lw=0.5)

    # Set x-axis ticks per month, including the most left one and most right one
    ax.xaxis.set_major_locator(mdates.MonthLocator())  # Place ticks at months

    # Make sure first and last values are included as ticks
    all_ticks = list(ax.get_xticks())
    first_date_num = mdates.date2num(pd.Timestamp(date[0]).to_pydatetime())
    last_date_num = mdates.date2num(pd.Timestamp(date[-1]).to_pydatetime())

    # Add first and last if not already close to existing ticks
    all_ticks.insert(0, first_date_num)
    all_ticks.append(last_date_num)
    ax.set_xticks(all_ticks)

    # Rotate x-axis ticks
    ax.tick_params(axis='x', rotation=90)

    # Format as YYYY.mm
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m'))

    ax.set_xlim([date[0], date[-1]])
    ax.grid(True, linestyle='--', alpha=0.7, lw=0.5)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(feat_name, fontsize=12)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
