import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import random

### Function Definitions ###

# Function to format biclusters into a list of sorted rows and columns
def format_biclusters(bicluster_list):
    """
    Format a list of biclusters by sorting their row and column indices.

    Parameters:
    bicluster_list (list): List of biclusters with 'rows' and 'cols' attributes.

    Returns:
    list: List of tuples, each containing sorted row and column indices of a bicluster.
    """
    formatted_biclusters = []
    for bicluster in bicluster_list:
        rows = sorted(bicluster.rows)  # Sort the row indices
        cols = sorted(bicluster.cols)  # Sort the column indices
        formatted_biclusters.append((list(rows), list(cols)))
    return formatted_biclusters

# Function to retrieve customers associated with a specific bicluster
def get_customers_by_bicluster(bicluster_name, df_groundtruth):
    """
    Retrieve customers belonging to a specific bicluster.

    Parameters:
    bicluster_name (str): Name of the bicluster.
    df_groundtruth (pd.DataFrame): DataFrame containing ground truth biclusters.

    Returns:
    pd.DataFrame: Filtered rows containing the specified bicluster.
    """
    return df_groundtruth[df_groundtruth['Biclusters'].apply(lambda x: bicluster_name in x)]

# Function to introduce sparsity into a data matrix
def sparsify_matrix(data, sparsity_intensity, convert_nonzero_to_one=False):
    """
    Introduce sparsity into the data matrix by setting a fraction of non-zero elements to zero.

    Parameters:
    data (np.ndarray): Original data matrix.
    sparsity_intensity (float): Fraction of non-zero elements to set to zero in each row.
    convert_nonzero_to_one (bool): If True, convert all non-zero elements to 1 before sparsifying.

    Returns:
    np.ndarray: The sparsified data matrix.
    """
    sparsified_data = data.copy()
    if convert_nonzero_to_one:
        sparsified_data[sparsified_data != 0] = 1

    for row_idx in range(sparsified_data.shape[0]):
        non_zero_indices = np.nonzero(sparsified_data[row_idx])[0]
        num_elements_to_zero = int(len(non_zero_indices) * sparsity_intensity)
        indices_to_zero = np.random.choice(non_zero_indices, num_elements_to_zero, replace=False)
        sparsified_data[row_idx, indices_to_zero] = 0

    return sparsified_data

# Function to introduce sparsity into a submatrix of the data matrix
def sparsify_submatrix(data, sparsity_intensity, row_start, row_end, col_start, col_end, convert_nonzero_to_one=False):
    """
    Introduce sparsity into a specific submatrix of the data matrix.

    Parameters:
    data (np.ndarray): Original data matrix.
    sparsity_intensity (float): Fraction of non-zero elements to set to zero in the submatrix.
    row_start (int): Starting index of rows to sparsify (inclusive).
    row_end (int): Ending index of rows to sparsify (exclusive).
    col_start (int): Starting index of columns to sparsify (inclusive).
    col_end (int): Ending index of columns to sparsify (exclusive).
    convert_nonzero_to_one (bool): If True, convert all non-zero elements to 1 before sparsifying.

    Returns:
    np.ndarray: The data matrix with the specified submatrix sparsified.
    """
    sparsified_data = data.copy()
    submatrix = sparsified_data[row_start:row_end, col_start:col_end]

    if convert_nonzero_to_one:
        submatrix[submatrix != 0] = 1

    for row_idx in range(submatrix.shape[0]):
        non_zero_indices = np.nonzero(submatrix[row_idx, :])[0]
        num_elements_to_zero = int(len(non_zero_indices) * sparsity_intensity)
        if num_elements_to_zero > 0:
            indices_to_zero = np.random.choice(non_zero_indices, num_elements_to_zero, replace=False)
            submatrix[row_idx, indices_to_zero] = 0

    sparsified_data[row_start:row_end, col_start:col_end] = submatrix
    return sparsified_data

# Function to extend product lists with category-based naming
def extend_products_with_category_name(original_products, target_count):
    """
    Extend product lists for each category to a target count by generating new names.

    Parameters:
    original_products (dict): Dictionary of categories and their product lists.
    target_count (int): Target number of products for each category.

    Returns:
    dict: Extended dictionary with additional products named using the category name.
    """
    extended_products = {}
    for category, products_list in original_products.items():
        extended_list = products_list.copy()
        additional_count = target_count - len(products_list)
        for i in range(1, additional_count + 1):
            new_product = f"{category}{i}"
            extended_list.append(new_product)
        extended_products[category] = extended_list
    return extended_products

### Main DataFrame Generation Process ###

def generate_dataframe():
    np.random.seed(42)

    # Define product categories
    products = {
        "Tech": ["iPhone", "SamsungPhone", "HuaweiPhone", "DellLaptop", "Xiaomi", "Macbook", "Matepad", "AsusLaptop", "AlienComputer", "MicrosoftSurface"],
        "Music": ["BoseSpeaker", "YamahaViolin", "FenderGuitar", "BroadwayTicket", "SonyHeadphones", "CasioKeyboard", "ShureMicrophone", "JBLSubwoofer", "RolandDrumSet", "SennheiserEarbuds"],
        "Household": ["LGTelevision", "IkeaSofa", "TempurPillow", "SealyBed", "DysonVacuum", "SamsungFridge", "WhirlpoolWasher", "PanasonicMicrowave", "TefalIron", "PhilipsBlender"],
        "Sports": ["AdidasBall", "GatoradeBeverage", "NikeShoes", "UnderArmourShirt", "WilsonTennisRacket", "PumaSocks", "ReebokShorts", "YonexBadmintonRacket", "SpeedoSwimwear", "AsicsRunningShoes"],
        "DailyUsage": ["DoveShampoo", "LorealHairSpray", "NeutrogenaFacialWash", "ColgateToothpaste", "OralBToothbrush", "NiveaLotion", "GilletteRazor", "PanteneConditioner", "ListerineMouthwash", "OlayCream"],
        "Fashion": ["ZaraShirt", "LevisJeans", "NikeSneakers", "RolexWatch", "RayBanSunglasses", "GucciBag", "PradaDress", "AdidasCap", "TommyHilfigerJacket", "CalvinKleinUnderwear"],
        "Food": ["DominosPizza", "CocaCola", "StarbucksCoffee", "KFCChicken", "SubwaySandwich", "BurgerKingWhopper", "Pepsi", "DunkinDonuts", "TacoBellTaco", "PizzaHutPasta"]
    }

    products = extend_products_with_category_name(products, 30)
    product_list = [product for category in products.values() for product in category]

    # Define bicluster parameters
    bicluster_params = [
        ((3500, 66), (0, 0), 0.9, products["Tech"][:30] + products["Music"][:6] + products["Food"][:30]),
        ((2800, 36), (2700, 24), 0.8, products["Tech"][-6:] + products["Music"][:30]),
        ((2000, 30), (5500, 60), 0.85, products["Household"][:30]),
        ((3500, 30), (7500, 90), 0.75, products["Sports"][:30]),
        ((2200, 36), (11000, 120), 0.7, products["DailyUsage"][:30] + products["Fashion"][:6]),
        ((3300, 36), (12700, 144), 0.8, products["DailyUsage"][-6:] + products["Fashion"][:30]),
        ((1900, 30), (16000, 180), 0.85, products["Food"][:30])
    ]

    # Generate data matrix
    data = np.zeros((17900, 210), dtype=int)
    bicluster_labels = {i: [] for i in range(17900)}
    for idx, ((rows, cols), (row_start, col_start), density, product_subset) in enumerate(bicluster_params):
        col_indices = [product_list.index(product) for product in product_subset]
        bicluster = (np.random.random(size=(rows, cols)) * 5).astype(int)
        for i, product in enumerate(product_subset):
            data[row_start:row_start + rows, col_indices[i]] = bicluster[:, i]
        for i in range(row_start, row_start + rows):
            bicluster_labels[i].append(idx)

    # Assign bicluster names
    bicluster_names = [
        "Tech Enthusiast",
        "Music Lover",
        "Home Comfort Seeker",
        "Sports Fanatic",
        "Daily Essentials Lover",
        "Fashion Forward",
        "Foodie"
    ]

    customer_bicluster_names = {i: [bicluster_names[label] for label in bicluster_labels[i]] for i in range(17900)}
    df = pd.DataFrame(data, columns=product_list)
    df_groundtruth = df.copy()
    df_groundtruth['Biclusters'] = df_groundtruth.index.map(customer_bicluster_names)

    # Sparsify data
    sparsified_data = sparsify_matrix(df.values, 0.8, convert_nonzero_to_one=False)
    sparsified_data = sparsify_submatrix(sparsified_data, 0.7, 0, 3501, 180, 211, convert_nonzero_to_one=False)
    df_sparsified = pd.DataFrame(sparsified_data, columns=df.columns)

    # Add noise
    low, high = 1, 5
    density = 0.002
    mask = np.random.rand(*df_sparsified.shape) < density
    noise = np.random.randint(low, high, size=mask.sum())
    df_noisy = df_sparsified.copy()
    df_noisy.values[mask] += noise

    # Visualize
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_noisy.values, cmap='viridis', annot=False)
    plt.title('Heatmap of the DataFrame')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.show()

    return df_noisy, df_groundtruth
