import csv
import wikipedia as wp
import pandas as pd
from lista_paesi import country_list

'''
In questo file creiamo il nostro data frame dove inseriamo i dati per ogni paese divisi in colonne.
'''

if __name__ == '__main__':

    '''
    Apertura dei file csv contenenti i tweet relativi alle query per i 37 paesi,
    e successiva creazione del dizionario avente come chiave il paese e come valore
    una lista con numero di tweet positivi, negativi e neutri.
    '''

    country_sentiment = {}
    for item in country_list:
        with open('{}.csv'.format(item[1]), 'r') as csvfile:
            positive, negative, neutral = 0, 0, 0
            l = csvfile.readlines()
            for i in l:
                if 'positive' in i:
                    positive += 1
                if 'negative' in i:
                    negative += 1
                if 'neutral' in i:
                    neutral += 1
        country_sentiment[item[1]] = [positive, negative, neutral]
    print(country_sentiment) # dizionario contenente i paesi e il numero di tweet pos, neg e neu

    '''
    Estrazione da una tabella su Wikipedia delle informazioni relative alla produzione
    di energia rinnovabile dei 37 paesi selezionati, le quali vengono inserite in una lista.
    '''

    html = wp.page("List_of_countries_by_renewable_electricity_production").html().encode("UTF-8")
    try:
        df = pd.read_html(html)[1]
    except IndexError:
        df = pd.read_html(html)[0]
    print(df.to_string())
    df.to_csv("dataset_production_re_energy.csv", sep=',')

    fin = open("dataset_production_re_energy.csv", 'r')
    l = fin.readlines()
    # print(l)
    clean_l, my_countries = [], []
    for item in l:
        clean_l.append(item.replace('%', '').split(','))
    # print(clean_l)
    for i in clean_l:
        for j in country_list:
            if ('South Korea' in j[1]) and ('Korea Rep' in i[1]): # passaggio necessario per non perdere informazioni realtive alla S.Korea
                my_countries.append([j[1], i[3], i[4], i[5]])
            if ('Czech Republic' in j[1]) and ('Czechia' in i[1]): # passaggio necessario per non perdere informazioni realtive alla R.Ceca
                my_countries.append([j[1], i[3], i[4], i[5]])
            if j[1] in i[1]:
                my_countries.append([j[1], i[3], i[4], i[5]])
    print(my_countries) # lista contenente le informazioni relative ai paesi

    '''
    Creazione dataset servendosi delle strutture dati create in precedenza contenenti le
    informazioni relative ai 37 paesi.
    '''

    with open('data_set.csv', 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(('Country', 'Total_GWh_prod', 'Total_RE_Gwh', 'RE_%_of_total',
                         'Positive', 'Negative', 'Neutral', 'Total'))

    for country in country_sentiment:
        for paese in my_countries:
            if paese[0] in country:
                with open('data_set.csv', 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow((paese[0], paese[1], paese[2], paese[3],
                                     country_sentiment[country][0], country_sentiment[country][1], country_sentiment[country][2],
                                     country_sentiment[country][0] + country_sentiment[country][1] + country_sentiment[country][2]))


