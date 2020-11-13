##!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[159]:


import streamlit as st
import random
import pandas as pd
import investpy

import numpy as np
from PIL import Image


# In[ ]:

hide_streamlit_style = """
            <style>
            
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


image = Image.open('Goal.png')  ### Cambia su web
st.sidebar.image(image, use_column_width=True)


# In[62]:

pagina = st.sidebar.selectbox("Pagina", ['Pianificatore', 'Modello di regressione Cape', 'Modello di regressione bonds']) #, 'Decumulo'

if pagina == 'Pianificatore':

    st.title('Pianificatore per obiettivi')

    # In[ ]:


    st.write('''###  ''')
    st.write('''### Portafogli predefiniti''')


    # In[157]:


    portafogli = pd.read_excel('portafogli.xlsx') ### Cambia su web C:\Users\user\Downloads\Mauro_app\
    portafogli = portafogli.set_index('ASSET ',1)
    # portafogli = portafogli.drop('Unnamed: 2',1)
    

    listadf = [list(portafogli['O.Temporale'].values)]
    for col in portafogli.columns[1:]:
        lista = []
        li = list(portafogli[col].values)
        for el in li:
            valore = str(round(el*100,2))+"%"
            lista.append(valore)
        listadf.append(lista)
    
    portafogli_ = pd.DataFrame(listadf, index=portafogli.columns, columns=portafogli.index)
    portafogli_


    st.write('''###  ''')
    st.write('''### Portafogli predefiniti: rappresentazione grafica''')

    composizione = portafogli[['BOND','COMM','CASH','EQUITY']]
    composizione = composizione*100
    st.bar_chart(composizione)

    # In[ ]:


    st.write('''###  ''')
    st.write('''### Seleziona i tuoi parametri''')


    # In[ ]:

    elencoportafogli = ['Liquidità']+list(portafogli.index)

    a1 = st.selectbox('Seleziona il portafoglio', elencoportafogli)
    a0 = st.number_input('Capitale iniziale', 0, 10000000,10000) 
    a3 = st.number_input('Obiettivo', 0, 10000000,10000)
    a4 = st.slider('Ipotesi di inflazione media %', 0,10, 2)
    a2 = st.slider('Orizzonte temporale in mesi', 0,200, 36)

    a4=a4/100

    # In[128]:


    ## 
    scelta = a1

    if scelta != 'Liquidità':

        mu = portafogli['REND.ATTESO'][scelta]
        mu = (mu+1)**(1/12)

        sigma = portafogli['''VOL.ATTESA'''][scelta]
        sigma = sigma/(12**(1/2))
    else:
        mu = None
        sigma=None

    inflazione = (a4+1)**(1/12)

    # In[151]:



    def montecarlo(start, mu, sigma):
        lista_serie = []

        for i1 in range(300):
            lista = [start]
            for i in range (a2):
                try:
                    rend = random.normalvariate(mu, sigma)
                except:
                    rend=1
                lista.append(rend)
            lista_serie.append(lista)

        df = pd.DataFrame(lista_serie)
        df = df.transpose()
        df = df.cumprod()
        return df


    # In[154]:

    
    df = montecarlo(a0,mu, sigma)

    # aggiungo la colonna obiettivo

    lista_ob = [a3]
    for i in range(a2):
        lista_ob.append(inflazione)
    df['Obiettivo']=lista_ob
    df['Obiettivo'] = df['Obiettivo'].cumprod()

    # Aggiungo il valore a scadena del versato

    lista_vers = [a0]
    for i in range(a2):
        lista_vers.append(inflazione)
    df_versato = pd.DataFrame(lista_vers, columns=['Versato'])
    df_versato['Versato'] = df_versato['Versato'].cumprod()


    ob_scad = df.tail(1).Obiettivo.values
    vers_scad = df_versato.tail(1).Versato.values
    # MOstro l'obiettivo reale a scadenza
    st.write('''###  ''')
    st.write('''### Obiettivo reale a scadenza del periodo selezionato''')

    
    # In[156]:
    obiettivo_scadenza = pd.DataFrame(ob_scad, columns = ['''Valore reale dell' obiettivo a scadenza'''], index=['Valore'])
    obiettivo_scadenza

    st.write('''###  ''')
    st.write('''### Rappresentazione grafica di 300 simulazioni''')


    df['index']= df.index
    df = df.set_index('index')
    df_ = np.log(df)
    st.line_chart(df)


    # ## Calcolo le probabilità ad un dato orizzonte

    # In[148]:


    obiettivo = a3
    rilevazione = a2

    campionamento = df.drop('Obiettivo',1).head(rilevazione).tail(1)


    campionamento_ = np.array(campionamento)


    st.write('''###  ''')
    st.write('''### Probabilità calcolate (termini nominali)''')

    proba = len(np.where(campionamento_>=obiettivo)[0])/3
    proba_in = len(np.where(campionamento_>=a0)[0])/3
    lista_ = [proba, proba_in]
    df_proba = pd.DataFrame(lista_, index =['Probabilità di raggiungere o superare il capitale obiettivo', 'Probabilità di mantenere o superare il versamento iniziale'], columns = ['Valori in percentuale'] )
    df_proba

    st.write('''###  ''')
    st.write('''### Probabilità calcolate (termini reali)''')

    probar = len(np.where(campionamento_>=ob_scad)[0])/3
    proba_inr = len(np.where(campionamento_>=vers_scad)[0])/3
    lista_r = [probar, proba_inr]
    df_proba_reale = pd.DataFrame(lista_r, index =['Probabilità di raggiungere o superare il capitale obiettivo', 'Prob. di mantenere o superare il valore reale del capitale'], columns = ['Valori in percentuale'] )
    df_proba_reale


    # # Ad ora le variabili da modificare sono: 
    # - media e varianza
    # - orizzonte temporale (rilevazione)
    # - importo iniziale 
    # 

    # In[138]:


    st.write('''###  ''')
    st.write('''### Statistiche sull' orizzonte selezionato ''')

    statistiche = campionamento.transpose().describe()
    statistiche = statistiche.drop(['count', 'std', 'min', 'max'],0)
    statistiche['statistiche'] = statistiche.values
    lista_ind = ["Risultato medio delle simulazioni nell' orizzonte temp.", "Risultato medio primo quartile", "Risultato medio secondo quartile", "Risultato medio terzo quartile"]
    statistiche = statistiche[['statistiche']]
    statistiche['indice']=lista_ind
    statistiche = statistiche.set_index('indice',1)
    statistiche


    st.write("""
    #  
     """)
    st.write("""
    ## DISCLAIMER:
     """)
    st.write("""
    Il contenuto del presente report non costituisce e non può in alcun modo essere interpretato come consulenza finanziaria, né come invito ad acquistare, vendere o detenere strumenti finanziari.
    Le analisi esposte sono da interpretare come supporto di analisi statistico-quantitativa e sono completamente automatizzate: tutte le indicazioni sono espressione di algoritmi matematici applicati su dati storici.
    Sebbene tali metodologie rappresentino modelli ampiamente testati e calcolati su una base dati ottenuta da fonti attendibili e verificabili non forniscono alcuna garanzia di profitto.
    In nessun caso il contenuto del presente report può essere considerato come sollecitazione all’ investimento. Si declina qualsiasi responsabilità legata all'utilizzo improprio di questa applicazione.
    I contenuti sono di proprietà di **Mauro Pizzini e Fabrizio Monge** e sia la divulgazione, come la riproduzione totale o parziale sono riservati ai sottoscrittori del servizio.
     """)
    
if pagina == 'Modello di regressione Cape':
    
    st.title('Modello di regressione su Shiller Cape')
    st.write('''###  ''')
    
    st.write('''###  ''')
    st.write('''### Inserire i parametri ''')
    proiezione = st.slider('proiezione in mesi', 60,240,120)
    
    # Import Data market

    import pandas as pd
    import pandas_datareader as pdr
    data = pd.DataFrame(pdr.get_data_yahoo('^GSPC', start='1-1-1990')['Close'])
    data = data.resample('M').last()


    import quandl
    quandl.ApiConfig.api_key = "sYKsCyP1pK54uPbEixD5"

    mydata = quandl.get("MULTPL/SHILLER_PE_RATIO_MONTH", start_date="1990-1-1")
    mydata = mydata.resample('M').last()

    mydata['PE_SHILLER']=mydata.Value.values
    mydata['Close']=data.Close.values[:len(mydata)]

    mydata = mydata.drop('Value',1)

    data = mydata
    import numpy as np
    data['REND FORWARD  -%-'] = (data.Close.shift(-proiezione)/data.Close-1)#np.log(data.Close.shift(-60)/data.Close)
    
    ## Build start and END
    data=data.reset_index()


    from datetime import date
    from dateutil.relativedelta import relativedelta

    lista=[]
    data['Start']=data['Date']
    for i in data['Start']:
        end_ = i+relativedelta(months=+proiezione)
        lista.append(end_)
    data['End']=lista
    data = data.set_index('Date',1)
    
    data['REND FORWARD  -%-']=round(data['REND FORWARD  -%-'],2)*100
    
    ## Build linear model
    #sklearn.linear_model.LinearRegression

    from sklearn.linear_model import LinearRegression
    lin = LinearRegression()
    X = data.dropna().PE_SHILLER.values.reshape(-1,1)
    y = data.dropna()['REND FORWARD  -%-'].values
    lin = lin.fit(X, y)

    #Predict
    X = data.PE_SHILLER.values.reshape(-1,1)
    data['Forecast -%-']=lin.predict(X)

    data_last=data.tail(1)
    
#     data_last
    anni = proiezione/12
    
    data_last['Forecast -%-  ANNUO'] = data_last['Forecast -%-']/anni
    
    data_last_exp = data_last[['Forecast -%-', 'Forecast -%-  ANNUO']]
    
    st.write('''###  ''')
    st.write('''### Tabella proiezione ''')
    data_last_exp
    
    st.write('''###  ''')
    st.write('''### Grafico del Modello di regressione Cape''')
    import altair as alt
    fig1 = alt.Chart(data).mark_circle(size=200).encode(alt.X('PE_SHILLER',scale=alt.Scale(zero=False)), y='REND FORWARD  -%-',tooltip=['Start', 'End','PE_SHILLER','REND FORWARD  -%-']).properties(height=500)
    fig2 = alt.Chart(data_last).mark_circle(size=200, color='red').encode(x='PE_SHILLER', y='Forecast -%-',tooltip=['Start', 'End','PE_SHILLER', 'Forecast -%-']).properties(height=500)
    regr = alt.Chart(data).mark_line(color='green').encode(x='PE_SHILLER',y='Forecast -%-' , size=alt.value(0.6))
    rule = alt.Chart(data_last).mark_rule(color = 'red', style='dotted').encode( x='PE_SHILLER',size=alt.value(0.6))
    immagine = fig1+fig2+rule+regr
    st.altair_chart(immagine, use_container_width=True)
    
    st.markdown('''## Cos'è il CAPE di SHILLER?
Il termine CAPE sta per cyclically-adjusted price-earnings ratio (Cape), ovvero rapporto prezzo utili aggiustato per i cicli, ed è stato creato dal professor Shiller. In termini molto semplici, viene utilizzato il valore di un indice che viene confrontato, invece che con l’ultimo andamento degli utili, con la media a 10 anni degli utili storici. La fotografia del mercato viene estesa a un arco temporale molto lungo capace di cogliere le varie fasi di mercato: valori bassi intorno a 10-15 indicano prospettive interessanti. Questo indicatore infatti viene utilizzato per formulare ipotesi di attesa dei rendimenti nei 10 successivi alla rilevazione.  Pur essendo impossibile ipotizzare con certezza i rendimenti futuri, lo strumento in passato ha fornito scenari realistici ed una buona correlazione con i rendimenti effettivi futuri.
''')
    
    st.markdown('''## L'analisi
Abbiamo importato i valori del Cape di Shiller dal 1990 e i valori dell' indice S&P500, rappresentativo delle maggiori 500 aziende quotate sul mercato americano.
Inseguito abbiamo calcolato i rendimenti delle finestre pluriennali e le abbiamo collegate al valore del PE di Shiller ad inizio periodo. Ad esempio, la finestra di 5 anni che va dal 31 gennaio 1990 al 31 gennaio 1995 è iniziata con un Shiller pe pari a 17.5 e ha avuto una performance del 43% nel periodo. Quindi si troverà un punto sul grafico che avrà come coordinate 17.5 (valore del PE) sull'asse orizzontale e 43 (la performance) sull' asse verticale.
Con finestre mobili di 5 anni, i punti sul grafico sono 370, perchè ad ogni mese corrisponde una finestra temporale che si estende per i successivi 5 anni, fino quella che inizia ad ottobre di 5 anni fa. Successivamente non ci sono dati perchè i 5 anni non sono ancora trascorsi.  ''')

    st.write("""
    #  
     """)
    st.write("""
    ## DISCLAIMER:
     """)
    st.write("""
    Il contenuto del presente report non costituisce e non può in alcun modo essere interpretato come consulenza finanziaria, né come invito ad acquistare, vendere o detenere strumenti finanziari.
    Le analisi esposte sono da interpretare come supporto di analisi statistico-quantitativa e sono completamente automatizzate: tutte le indicazioni sono espressione di algoritmi matematici applicati su dati storici.
    Sebbene tali metodologie rappresentino modelli ampiamente testati e calcolati su una base dati ottenuta da fonti attendibili e verificabili non forniscono alcuna garanzia di profitto.
    In nessun caso il contenuto del presente report può essere considerato come sollecitazione all’ investimento. Si declina qualsiasi responsabilità legata all'utilizzo improprio di questa applicazione.
    I contenuti sono di proprietà di **Mauro Pizzini e Fabrizio Monge** e sia la divulgazione, come la riproduzione totale o parziale sono riservati ai sottoscrittori del servizio.
     """)

if pagina == 'Modello di regressione bonds':


    st.title('Modello di regressione su rendimenti bonds')
    st.write('''###  ''')

    proiezioni = 120

    df = investpy.get_bond_historical_data(bond='U.S. 10Y', from_date='01/01/2000', to_date='14/10/2020', interval='Monthly')[['Close']]
    df = pd.DataFrame(df.values, index=df.index, columns=['Tasso di rendimento'])
    df_ = investpy.get_index_historical_data(index='TR US 10 Year Government Benchmark', country='united states', from_date='01/01/2000', to_date='14/10/2020', interval='Monthly')[['Close']]
    df_ = pd.DataFrame(df_.values, columns=['prezzo'], index=df_.index)       

    df = df.join(df_)

    df = df.resample('M').last()
    df = df.fillna(method='ffill')

    # Build start and end period

    df=df.reset_index()


    from datetime import date
    from dateutil.relativedelta import relativedelta

    lista=[]
    df['Start']=df['Date']
    for i in df['Start']:
        end_ = i+relativedelta(months=+proiezioni)
        lista.append(end_)
    df['End']=lista
    df = df.set_index('Date',1)

    # Build forward

    df['Forward']= (df.prezzo.shift(-proiezioni)/df.prezzo-1)*100

    # Build linear model for prediction

    from sklearn.linear_model import LinearRegression
    lin = LinearRegression()
    X = df.dropna()['Tasso di rendimento'].values.reshape(-1,1)
    y = df.dropna()['Forward'].values
    lin = lin.fit(X, y)

    #Predict
    X = df['Tasso di rendimento'].values.reshape(-1,1)
    df['Forecast -%-']=lin.predict(X)

    df_last=df.tail(1)

    # Plot interactive

    import altair as alt
    fig1 = alt.Chart(df).mark_circle(size=200).encode(alt.X('Tasso di rendimento',scale=alt.Scale(zero=False)), y='Forward',tooltip=['Start', 'End','Tasso di rendimento','Forward']).properties(height=500)
    fig2 = alt.Chart(df_last).mark_circle(size=200, color='red').encode(x='Tasso di rendimento', y='Forecast -%-',tooltip=['Start', 'End','Tasso di rendimento', 'Forecast -%-']).properties(height=500)
    regr = alt.Chart(df).mark_line(color='green').encode(x='Tasso di rendimento',y='Forecast -%-' , size=alt.value(0.6))
    rule = alt.Chart(df_last).mark_rule(color = 'red', style='dotted').encode( x='Tasso di rendimento',size=alt.value(0.6))
    immagine2 = fig1+fig2+rule+regr

    st.write('''###  ''')
    st.write('''### Tabella proiezione Treasury 10Y''')

    df_last_proiezione = df_last[['Forecast -%-']]
    df_last_proiezione['Forecast -%- ANNUO'] = df_last_proiezione['Forecast -%-']/10

    df_last_proiezione



    st.write('''###  ''')
    st.write('''### Grafico del Modello di regressione - Treasury 10Y''')

    st.altair_chart(immagine2, use_container_width=True)

if pagina == 'Decumulo':

    st.title('Pianificatore per obiettivi')

    # In[ ]:


    st.write('''###  ''')
    st.write('''### Portafogli predefiniti''')


    # In[157]:


    portafogli = pd.read_excel(r'C:\Users\user\Downloads\Mauro_app\portafogli.xlsx') ### Cambia su web
    portafogli = portafogli.set_index('ASSET ',1)
    # portafogli = portafogli.drop('Unnamed: 2',1)
    

    listadf = [list(portafogli['O.Temporale'].values)]
    for col in portafogli.columns[1:]:
        lista = []
        li = list(portafogli[col].values)
        for el in li:
            valore = str(round(el*100,2))+"%"
            lista.append(valore)
        listadf.append(lista)
    
    portafogli_ = pd.DataFrame(listadf, index=portafogli.columns, columns=portafogli.index)
    portafogli_


    st.write('''###  ''')
    st.write('''### Portafogli predefiniti: rappresentazione grafica''')

    composizione = portafogli[['BOND','COMM','CASH','EQUITY']]
    composizione = composizione*100
    st.bar_chart(composizione)

    # In[ ]:


    st.write('''###  ''')
    st.write('''### Seleziona i tuoi parametri''')


    # In[ ]:


    a1 = st.selectbox('Seleziona il portafoglio', list(portafogli.index))
    a0 = st.number_input('Capitale iniziale', 10000) 
    a3 = st.number_input('Rendita finanziaria mensile',0,100000, 1000)
    a2 = st.slider('Periodo in cui verrà erogata la rendita', 0,200, 60)


    # In[128]:


    ## 
    scelta = a1
    mu = portafogli['REND.ATTESO'][scelta]
    mu = (mu+1)**(1/12)

    sigma = portafogli['''VOL.ATTESA'''][scelta]
    sigma = sigma/(12**(1/2))


    # In[151]:



    def montecarlo_rendita(start, mu, sigma, rendita):
        lista_serie = []

        for i1 in range(300):
            lista = [start]
            for i in range (a2):
                rend = random.normalvariate(mu, sigma)
                lista.append(rend)
            lista_serie.append(lista)

        df = pd.DataFrame(lista_serie)
        df = df.transpose()




        df = df.cumprod()
        return df


    # In[154]:


    df = montecarlo(a0,mu, sigma)


    # In[156]:


    st.write('''###  ''')
    st.write('''### Rappresentazione grafica di 300 simulazioni''')


    df['index']= df.index
    df = df.set_index('index')
    df_ = np.log(df)
    

    df_rendita = pd.DataFrame(index=df.index, columns=df.columns)
    lista=list(df.index)
    for i in df_rendita.columns:
        df_rendita[i]=lista

    df_rendita = df_rendita*a3
    

    df = (df-df_rendita)
    df = df[df>0]
    df = df.fillna(0)
    
    st.line_chart(df)
