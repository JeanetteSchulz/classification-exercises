import pandas as pd
import numpy as np
import os

"""
This file is for the Codeup example dataframes used in the SQL database. This file saves functions
to be resued as necessary, so that this code does not need to be copied or rewritten. These
functions read the Titanic, Iris, and Telco data sets from the Codeup database. These functions indivdually read the 
databases into a variable, and then clean and split them. Each function works only for it's individually named dataset 
and can NOT be interchanged, since the cleaning process for each is unquie to the individual database.
"""

###################### Acquire Titanic Data ######################

def get_titanic_data():
    '''
    This function reads in titanic data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('titanic_df.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('titanic_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = pd.read_sql('SELECT * FROM passengers', get_db_url('titanic_db'))
    
        # Write DataFrame to a csv file.
        df.to_csv('titanic_df.csv')
        
    return df

###################### Acquire Iris Data ######################

def get_iris_data():
    '''
    This function reads in iris data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('iris_df.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('iris_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = pd.read_sql("""SELECT * FROM measurements JOIN species USING(species_id);""" , get_db_url("iris_db"))
        
        # Cache data
        df.to_csv('iris_df.csv')
        
    return df

###################### Acquire Telco Data ######################

def get_telco_data():
    '''
    This function reads in iris data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('telco.csv'):
        # If csv file exists read in data from csv file.
        df = pd.read_csv('telco.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df =  pd.read_sql("""SELECT * FROM customers
                            JOIN contract_types USING(contract_type_id)
                            JOIN internet_service_types USING(internet_service_type_id)
                            JOIN payment_types USING(payment_type_id)""", 
                            get_db_url('telco_churn')
                        )
        
        # Cache data
        df.to_csv('telco.csv')
        
    return df

###################### Clean Titanic Data ######################

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def prep_titanic (titanic):
    """
    Boop
    """
    # Dropping duplicates (if any)
    titanic = titanic.drop_duplicates()

    # Dropping columns with too many missing values (like we saw in the lesson)
    titanic = titanic.drop(['deck', 'embarked', 'class', 'age'], 1)

    # embark_town has some NULLs, I'm gonna run .fillna() on the column with the most popular port
    titanic['embark_town'] = titanic.embark_town.fillna(value='Southampton')

    # Making some dummies!
    titanic_dummy = pd.get_dummies(titanic[['sex', 'embark_town']], dummy_na=False, drop_first= False )

    # I only want one column for gender, but I want to keep all embark dummies for now
    titanic_dummy = titanic_dummy.drop('sex_male', 1)
    titanic_dummy = titanic_dummy.rename(columns={"sex_female": "is_female"})

    # Okay, now to concatenate my dummy dataframe to my titanic dataframe
    titanic = pd.concat([titanic, titanic_dummy], axis=1)
    
    # Splitting the data for testing!
    train, test = train_test_split(titanic, test_size = .2, random_state=22, stratify=titanic.survived)
    train, validate = train_test_split(train, test_size=.3, random_state=22, stratify=train.survived)
   
    imputer = SimpleImputer(missing_values = np.nan, strategy='most_frequent')
    train[['embark_town']] = imputer.fit_transform(train[['embark_town']])
    validate[['embark_town']] = imputer.transform(validate[['embark_town']])
    test[['embark_town']] = imputer.transform(test[['embark_town']])
    
    return train, validate, test


###################### Clean Iris Data ######################

def prep_iris(iris):
    """
    Beep
    """
    iris = iris.drop(["species_id","measurement_id"], 1)
    iris = iris.rename(columns={"species_name": "species"})
    iris_dummy = pd.get_dummies(iris[['species']], dummy_na=False )
    iris = pd.concat([iris, iris_dummy], axis=1)
    
    # Splitting the data for testing!
    train, test = train_test_split(iris, test_size = .2, random_state=22, stratify= iris.species)
    train, validate = train_test_split(train, test_size=.3, random_state=22, stratify= train.species)
    return train, validate, test

###################### Clean Telco Data ######################

def prep_telco(telco):
    """
    Booop
    """
    # Some total_charges are blank. Let's convert those to zero for now, or I won't be able to change dtype
    telco = telco.assign(total_charges = telco.total_charges.replace(" ", "0.00"))

    # Total_Charges needs to be changed from Object to a Float
    telco["total_charges"]= telco["total_charges"].str.strip().replace(",","").replace("$","").astype(float)

    # Always Drop them duplicates (if any)
    telco = telco.drop_duplicates()

    # Time to make a LOT of dummies for all the categorical columns
    telco_dummy1 = pd.get_dummies(
                                  telco[[ 'gender', 
                                          'partner', 
                                          'dependents', 
                                          'phone_service', 
                                          'paperless_billing',
                                          'churn',
                                          'multiple_lines', 
                                          'online_security', 
                                          'online_backup', 
                                          'device_protection', 
                                          'tech_support',
                                          'streaming_tv']], 
                                          dummy_na= False, drop_first= True
                                 )
    # Making two dummies because it's easier than dropping a bunch of columns after
    telco_dummy2 = pd.get_dummies(
                                  telco[[ 
                                          'internet_service_type',
                                          'payment_type',
                                          'contract_type']], 
                                          dummy_na= False, #drop_first= True
                                 )

    # Now to concatenate my dummy dataframes to my telco dataframe
    telco = pd.concat([telco, telco_dummy1, telco_dummy2], axis=1)
    telco.head()

    # Looks great! I'm gonna rename some columns for clarity tho:
    telco = telco.rename(columns={"gender_Male": "is_male"})
    telco = telco.rename(columns={"partner_Yes": "has_partner"})
    telco = telco.rename(columns={"dependents_Yes": "has_dependent"})
    telco = telco.rename(columns={"phone_service_Yes": "has_phone_service"})
    telco = telco.rename(columns={"paperless_billing_Yes": "has_paperless_billing"})
    telco = telco.rename(columns={"churn_Yes": "has_churned"})
    telco = telco.rename(columns={"contract_type_One year": "one_year_contract"})
    telco = telco.rename(columns={"contract_type_Two year": "two_year_contract"})
    telco = telco.rename(columns={"multiple_lines_Yes": "has_multiple_lines"})
    telco = telco.rename(columns={"multiple_lines_No phone service": "multiple_lines_no_phone_service"})
    telco = telco.rename(columns={"online_security_No internet service": "online_security_no_internet_service"})
    telco = telco.rename(columns={"online_security_Yes": "has_online_security"})
    telco = telco.rename(columns={"online_backup_No internet service": "online_backup_no_internet_service"})
    telco = telco.rename(columns={"online_backup_Yes": "has_online_backup"})
    telco = telco.rename(columns={"device_protection_No internet service": "device_protection_no_internet_service"})
    telco = telco.rename(columns={"device_protection_Yes": "has_device_protection"})
    telco = telco.rename(columns={"tech_support_No internet service": "tech_support_no_internet_service"})
    telco = telco.rename(columns={"tech_support_Yes": "has_tech_support"})
    telco = telco.rename(columns={"streaming_tv_No internet service": "streaming_tv_no_internet_service"})
    telco = telco.rename(columns={"streaming_tv_Yes": "has_streaming_tv"})
    telco = telco.rename(columns={"internet_service_type_Fiber optic": "internet_service_type_fiber_optic"})
    telco = telco.rename(columns={"payment_type_Bank transfer (automatic)": "payment_type_bank_transfer"})
    telco = telco.rename(columns={"payment_type_Credit card (automatic)": "payment_type_credit_card"})
    telco = telco.rename(columns={"payment_type_Electronic check": "payment_type_electronic_check"})
    telco = telco.rename(columns={"payment_type_Mailed check": "payment_type_mailed_check"})
    telco = telco.rename(columns={"contract_type_Month-to-month": "month_to_month_contract"})

    # Splitting the data for testing!
    train, test = train_test_split(telco, test_size = .2, random_state=22, stratify= telco.has_churned)
    train, validate = train_test_split(train, test_size=.3, random_state=22, stratify= train.has_churned)
    return train, validate, test

