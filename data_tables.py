import csv
import wikipedia as wp
import pandas as pd
from lista_paesi import country_list

if __name__ == '__main__':

    '''
    Estrazione da una tabella su Wikipedia delle informazioni relative al consumo
    di energia elettrica dei 37 paesi selezionati, le quali vengono inserite in una lista.
    '''

    html = wp.page("List_of_countries_by_electricity_consumption").html().encode("UTF-8")
    try:
        df = pd.read_html(html)[2]
    except IndexError:
        df = pd.read_html(html)[0]
    print(df.to_string())
    df.to_csv("dataset_electricity_consumption.csv", sep=',')

    fin = open("dataset_electricity_consumption.csv", 'r')
    l2= fin.readlines()
    # print(l2)
    clean_l2, countries = [], []
    for ktem in l2:
        clean_l2.append(ktem.replace('%', '').replace('Korea, South', 'South Korea').split(','))
    # print(clean_l2)
    for i in clean_l2:
        for j in country_list:
            if j[1] in i[2]:
                countries.append([j[1], i[3], i[6], i[8]])
    print(countries)

    '''
    Estrazione da una tabella su Wikipedia delle informazioni relative alle emissioni
    di anidride carbonica dei 37 paesi selezionati, le quali vengono inserite in una lista.
    '''

    html = wp.page("List_of_countries_by_carbon_dioxide_emissions").html().encode("UTF-8")
    try:
        df = pd.read_html(html)[1]
    except IndexError:
        df = pd.read_html(html)[0]
    print(df.to_string())
    df.to_csv("dataset_carbon_d_emissions.csv", sep=',')

    fin3 = open("dataset_carbon_d_emissions.csv", 'r')
    l3 = fin3.readlines()
    # print(l3)
    clean_l3, countries3 = [], []
    for item in l3:
        clean_l3.append(item.replace('%', '').split(','))
    # print(clean_l3)
    for i in clean_l3:
        for j in country_list:
            if j[1] in i[1]:
                countries3.append([j[1], i[4], i[6]])
    print(countries3)

    '''
    Creazione del secondo file csv, in cui vengono inserite le informazioni relative alle tabelle
    estratte su electricity consumption e CO2 emissions per i 37 paesi.
    '''

    with open('data_set2.csv', 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(("Country", "Electricity Consumption", "Population", "Avg Consumption per capita", "Fossil CO2 Emissions",
       "Fossil CO2 Emissions: 2017vs1990, change(%)"))

    for j in countries3:
        for i in countries:
            if j[0] == i[0]:
                with open('data_set2.csv', 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow((i[0], i[1],  i[2], i[3], j[1],  j[2]))


