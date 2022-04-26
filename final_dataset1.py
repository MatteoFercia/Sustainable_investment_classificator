import pandas as pd

if __name__ == '__main__':

    '''
    Aggiunta al csv finale, per poter eseguire le analisi statistiche, della variabile relativa
    alla quota di CO2 emissions per capita per ciascuno dei 37 paesi e della response value, 
    che è stata ricavata assumendo che se la percentuale di energia green prodotta in quel paese
    è maggiore del 50% rispetto al totale dell'energia prodotta allora l'investimento è conveniente
    e si codifica il paese come Positive, altrimenti Negative.
    '''

    # APERTURA DEL FILE IN LETTURA E CREAZIONE LISTA PROVVISORIA SU CUI LAVORARE
    fin = open('dataset_finale.csv', 'r')
    l = fin.readlines()
    # print(l)
    clean_l, investimento, fossil_co2_emission_percapita = [], [], []
    for item in l:
        clean_l.append(item.split(','))
    # print(clean_l)

    # CREZIONE DELLE LISTE CONTENENTI LE INFORMAZIONI PER CIASCUN PAESE
    for item in clean_l[1:]:
        if float(item[3])/100 > .5:
            investimento.append('Positive')
        else:
            investimento.append('Negative')
        fossil_co2_emission_percapita.append(float(item[11])/float(item[9]))
    print(investimento)
    print(fossil_co2_emission_percapita)

    # AGGIUNTA DELLE DUE VARIABILI AL CSV
    df = pd.read_csv("dataset_finale.csv")
    df["Risultato Investimento"] = investimento
    df["Fossil_CO2_percapita"] = fossil_co2_emission_percapita
    df.to_csv("dataset_finale.csv", index=False)
