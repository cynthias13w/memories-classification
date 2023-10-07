from config import PATH_TO_DATA
import pandas as pd

def import_data(file_path):
    """
    Import data from a CSV file and return it as a Pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the imported data.
    """
    try:
        # Use Pandas to read the CSV file into a DataFrame
        data = pd.read_csv(file_path, sep= ";")

        # Return the DataFrame
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None
      