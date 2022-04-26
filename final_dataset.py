import pandas as pd

if __name__ == '__main__':

    '''
    Merge dei diversi dataset ottenuti per andare a creare quello su cui
    baseremo le analisi statistiche.
    '''
    # DATASET CONTENENTE LE INFORMAZIONI RELATIVE AI TWEET, AL LORO SENTIMENT, E ALLA ENERGY PRODUCTION
    dataset_1 = pd.read_csv("data_set.csv")
    # DATASET CONTENENTE LE INFORMAZIONI RELATIVE AL CONSUMO DI ENERGIA ELETTRICA E ALLE EMISSIONI DI CO2
    dataset_2 = pd.read_csv("data_set2.csv")
    # DATASET ESTRATTO DAL SITO DI OECD
    dataset_3 = pd.read_csv("green_investment.csv")
    # PRIMO MERGE
    merged = dataset_1.merge(dataset_2, on='Country')
    merged.to_csv("data_set_merged.csv", index=False)
    # SECONDO MERGE, CREAZIONE DATASET FINALE
    dataset_4 = pd.read_csv("data_set_merged.csv")
    merged = dataset_4.merge(dataset_3, on='Country')
    merged.to_csv("dataset_finale.csv", index=False)

