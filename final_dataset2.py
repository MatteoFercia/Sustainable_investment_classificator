from lista_paesi import country_list
import pandas as pd

def ret_investment(lista, numero1, numero2):
    '''
    Calcoliamo la quota di investimento green per ciascun paese
    '''
    investimento_green = []
    for item in lista:
        investimento_green.append([item[0],(float(item[2]) / numero2) * numero1])
    return investimento_green

def ret_clean_list(list):
    '''
    Restitiusce la lista senza i valori mancanti della tabella originale
    '''
    new_list = []
    for item in list:
        if item[2] == '':
            new_list.append([item[0], item[1], item[2].replace('', '0'), item[3]])
        else:
            new_list.append([item[0], item[1], item[2], item[3]])
    return new_list

def ret_total_re_prod(list):
    '''
    Calcolo produzione totale di energia per regione
    '''
    total_re = 0
    for item in list:
        total_re += float(item[2])
    return total_re


if __name__== '__main__':

    '''
    In questo script andiamo a ottenere informazioni relative agli investimenti dei paesi situati 
    all'interno delle regioni identificate dal database di Irena, per poi selezionare solo quelle relative
    ai 37 paesi scelti per la nostra analisi che saranno inserite all'interno di una lista la quale verrà
    aggiunta al dataset finale
    '''

    # CREAZIONE LISTE CON I PAESI RELATIVI ALLE REGIONI DEL DATABASE DI "IRENA"
    e_europe_c_asia = ['Albania', 'Armenia', 'Azerbaijan', 'Belarus', 'Bosnia and Herzegovina', 'Hungary', 'Turkey',
                       'Georgia', 'Kazakhstan', 'Kosovo', 'Kyrgyzstan', 'Moldova', 'Montenegro', 'North Macedonia',
                       'Estonia', 'Lithuania', 'Latvia', 'Israel', 'Serbia', 'Tajikistan', 'Turkmenistan', 'Ukraine', 'Uzbekistan']

    oecd_america = ['United States', 'Canada']

    w_europe = ['Austria', 'Belgium', 'Czech Republic', 'France', 'Germany', 'Italy', 'Spain', 'Portugal', 'Greece',
                'Denmark', 'Sweden', 'Iceland', 'Finland', 'Norway', 'Ireland', 'Liechtenstein', 'Luxembourg', 'Monaco', 'Netherlands',
                'Slovenia', 'Slovakia', 'Switzerland', 'United Kingdom']

    oecd_oceania = ['Australia', 'New Zealand', 'Papua New Guinea', 'Fiji', 'Solomon Islands', 'Vanuatu', 'New Caledonia', 'French Polynesia',
                    'Samoa', 'Guam', 'Kiribati', 'Federated States of Micronesia', 'Tonga', 'American Samoa', 'Northen Mariana Islands', 'Marshall Islands',
                    'Palau', 'Cook Islands', 'Wallis and Furtuna', 'Tuvalu', 'Nauru']

    east_asia_pacific = ['China', 'Hong Kong', 'Japan', 'Macau', 'Mongolia', 'South Korea', 'Taiwan',
                         'Northern Mariana Islands', 'Federated States of Micronesia', 'Fiji', 'French Polynesia', 'Kiribati',
                         'Marshall Islands', 'Nauru', 'New Caledonia', 'Palau', 'Solomon Islands', 'Tonga', 'Tuvalu', 'Vanuatu', 'Wallis and Futuna']

    latin_america_carribeans = ['Antigua & Barbuda', 'Aruba', 'Bahamas', 'Barbados', 'Cayman Islands', 'Cuba', 'Dominica', 'Dominican Republic',
                                'Grenada', 'Guadeloupe', 'Haiti', 'Jamaica', 'Martinique', 'Puerto Rico', 'Saint Barthélemy', 'St Kitts and Nevis', 'St Lucia', 'St Vincent and the Grenadines',
                                'Trinidad and Tobago', 'Turks and Caicos Islands', 'Virgin Islands', 'Belize', 'Costa Rica', 'El Salvador', 'Guatemala', 'Honduras', 'Mexico', 'Nicaragua', 'Panama',
                                'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'French Guiana', 'Guyana', 'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela',]

    # APERTURA CSV RELATIVO ALLA TABELLA SULLA PRODUZIONE DI ENERGIA
    fin = open('dataset_production_re_energy.csv', 'r')
    l = fin.readlines()
    clean_l = []
    for i in l:
        clean_l.append(i.split(','))
    # print(clean_l)

    # CREAZIONE LISTE CON VALORI DELLA PRODUZIONE DI ENERGIA PER CIASCUNA REGIONE
    l1, l2, l3, l4, l5, l6 = [], [], [], [], [], []

    # East Asia and pacific:
    for item in clean_l[1:]:
        for jtem in east_asia_pacific:
            if (jtem == 'South Korea') and (item[1] == 'Korea Rep'):
                l1.append([jtem, item[3], item[4], item[5]])
            if jtem in item[1]:
                l1.append([jtem, item[3], item[4], item[5]])
    # print(l1)

    # OECD America:
    for item in clean_l[1:]:
        for jtem in oecd_america:
            if jtem in item[1]:
                l2.append([jtem, item[3], item[4], item[5]])
    # print(l2)

    # Western Europe:
    for item in clean_l[1:]:
        for jtem in w_europe:
            if (jtem == 'Czech Republic') and (item[1] == 'Czechia'):
                l3.append([jtem, item[3], item[4], item[5]])
            if jtem in item[1]:
                l3.append([jtem, item[3], item[4], item[5]])
    # print(l3)

    # Latin America and The Carribeans:
    for item in clean_l[1:]:
        for jtem in latin_america_carribeans:
            if jtem in item[1]:
                l4.append([jtem, item[3], item[4], item[5]])
    # print(l4)

    # OECD Oceania:
    for item in clean_l[1:]:
        for jtem in oecd_oceania:
            if jtem in item[1]:
                l5.append([jtem, item[3], item[4], item[5]])
    # print(l5)

    # Eastern Europe and Central Asia:
    for item in clean_l[1:]:
        for jtem in e_europe_c_asia:
            if jtem in item[1]:
                l6.append([jtem, item[3], item[4], item[5]])
    # print(l6)

    # PULIZIA LISTE DAI VALORI MANCANTI NELLA TABELLA ORIGINALE
    # East Asia and Pacific
    clean_l1 = ret_clean_list(list=l1)
    # print(clean_l1)

    # OECD America
    clean_l2 = ret_clean_list(list=l2)
    # print(clean_l2)

    # Western Europe
    clean_l3 = ret_clean_list(list=l3)
    # print(clean_l3)

    # Latin America and The Carribeans:
    clean_l4 = ret_clean_list(list=l4)
    # print(clean_l4)

    # OECD Oceania:
    clean_l5 = ret_clean_list(list=l5)
    # print(clean_l5)

    # Central Asia and Eastern Europe:
    clean_l6 = ret_clean_list(list=l6)
    # print(clean_l6)

    # QUOTE DI INVESTIMENTI TOTALI PER REGIONE (IN MILIARDI DI $)
    # Dati estratti dal database di "IRENA"
    inv_east_asia = 92.815769
    inv_oecd_america = 51.47268605
    inv_west_eu = 54.616582
    inv_latin = 10.339868
    inv_oecd_oceania = 4.428495
    inv_central_asia_east_eu = 2.667894

    # CALCOLO TOTALE PRODUZIONE DI ENERGIA PER CIASCUNA REGIONE
    # East Asia and Pacific
    tot_re_east_asia = ret_total_re_prod(list=clean_l1)
    # print(tot_re_east_asia)

    # OECD America:
    tot_re_oecd_america = ret_total_re_prod(list=clean_l2)
    # print(tot_re_oecd_america)

    # Western Europe:
    tot_re_west_eu = ret_total_re_prod(list=clean_l3)
    # print(tot_re_west_eu)

    # Latin America and The Carribeans:
    tot_re_latin = ret_total_re_prod(list=clean_l4)
    # print(tot_re_latin)

    # OECD Oceania:
    tot_re_oecd_oceania = ret_total_re_prod(list=clean_l5)
    # print(tot_re_oecd_oceania)

    # Central Asia and Eastern Europe:
    tot_re_central_asia_east_eu = ret_total_re_prod(list=clean_l6)
    # print(tot_re_central_asia_east_eu)

    # CALCOLO QUOTA DI INVESTIMENTI PER I PAESI SITUATI NELLE REGIONI
    inv_re_paese_east_asia = (ret_investment(lista=clean_l1, numero1=inv_east_asia, numero2=tot_re_east_asia))
    # print(inv_re_paese_east_asia)
    inv_re_paese_oecd_america = (ret_investment(lista=clean_l2, numero1=inv_oecd_america, numero2=tot_re_oecd_america))
    # print(inv_re_paese_oecd_america)
    inv_re_paese_west_eu = (ret_investment(lista=clean_l3, numero1=inv_west_eu, numero2=tot_re_west_eu))
    # print(inv_re_paese_west_eu)
    inv_re_paese_latin = (ret_investment(lista=clean_l4, numero1=inv_latin, numero2=tot_re_latin))
    # print(inv_re_paese_latin)
    inv_re_paese_oecd_oceania = (ret_investment(lista=clean_l5, numero1=inv_oecd_oceania, numero2=tot_re_oecd_oceania))
    # print(inv_re_paese_oecd_oceania)
    inv_re_paese_central_asia_east_eu = (ret_investment(lista=clean_l6, numero1=inv_central_asia_east_eu, numero2=tot_re_central_asia_east_eu))
    # print(inv_re_paese_central_asia_east_eu)

    # CREAZIONE LISTA PROVVISORIA CON I 37 PAESI SCELTI E QUOTA DI INVESTIMENTI
    temp_list, col_to_csv = [], []
    for item in country_list:
        for jtem in inv_re_paese_east_asia:
            if item[1] == jtem[0]:
                temp_list.append([item[1], jtem[1]])
    for item in country_list:
        for jtem in inv_re_paese_oecd_america:
            if item[1] == jtem[0]:
                temp_list.append([item[1], jtem[1]])
    for item in country_list:
        for jtem in inv_re_paese_west_eu:
            if item[1] == jtem[0]:
                temp_list.append([item[1], jtem[1]])
    for item in country_list:
        for jtem in inv_re_paese_latin:
            if item[1] == jtem[0]:
                temp_list.append([item[1], jtem[1]])
    for item in country_list:
        for jtem in inv_re_paese_oecd_oceania:
            if item[1] == jtem[0]:
                temp_list.append([item[1], jtem[1]])
    for item in country_list:
        for jtem in inv_re_paese_central_asia_east_eu:
            if item[1] == jtem[0]:
                temp_list.append([item[1], jtem[1]])

    print(temp_list)

    # SORT MANUALE DEI PAESI PER AVERLI NELLO STESSO ORDINE DEL CSV PER FACILITARE L'AGGIUNTA DELLA COLONNA
    for i in country_list:
        for k in temp_list:
            if i[1] == k[0]:
                col_to_csv.append(k[1])
    print(col_to_csv)

    # AGGIUNTA COLONNA AL DATASET
    df = pd.read_csv("dataset_finale.csv")
    df["Investimenti Green"] = col_to_csv
    df.to_csv("dataset_finale.csv", index=False)
