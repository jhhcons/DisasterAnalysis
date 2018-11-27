import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ Load message and category file, merge by id and return df
    
    Arguments:
    messages_filepath -- filepath to messages csv
    categories_filepath -- filepath to categories csv
 
    Return Values:
    df -- Loaded df
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)    
    return messages.merge(categories, on='id')

def clean_data(df):
    """ Create new categories df and remove dublicates
    The categories column is a string with the  format 'related-1;request-0;...'
    It is desired to convert this into 36 new columns with a 1 or 0 value    
    
    Arguments:
    df -- dataframe to clean
 
    Return Values:
    df -- Clean dataframe
    """
    
    # Split category string by ';'
    categories = df["categories"].str.split(";", expand=True)
    # Fetch names for columns from first row
    row = categories.iloc[0,:]
    # Get columnames by selected up to the second-last character
    category_colnames = row.str.slice(0,-2)
    categories.columns = category_colnames
    
    # Extract values of the columns
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(-1)

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # Replace columns string with actual columns
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    
    # Remove any dublicates
    df = df[~df.duplicated()]
    return df

def save_data(df, database_filename):
    """ Save dataframe to sql databse
    
    Arguments:
    df -- dataframe to save
    database_filename -- path to database
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterTable', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()