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


image = Image.open('Annuity.png')  ### Cambia su web C:\Users\user\Downloads\Mauro_app\
st.sidebar.image(image, use_column_width=True)


# In[62]:

pagina = st.sidebar.selectbox("Pagina", ['Applicazione', 'Tutorials e Formazione', 'Video Tutorial'])


if pagina == 'Applicazione':

    st.title('Calcolatore grafico di rendita finanziaria')

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
            
    
    a0 = st.number_input('Capitale iniziale', 1000000) 
    a3 = st.number_input('Rendita finanziaria mensile',0,100000, 4000)
    a4 = st.slider('''% di adeguamento all' inflazione''',0,10,2)
            
    multi = st.checkbox('Multi periodo')

    if multi == False:

        a22 = 0
        a2 = st.slider('Periodo in cui verrà erogata la rendita (anni)', 0,40, 20)
        a1 = st.selectbox('Seleziona il portafoglio', list(portafogli.index))
        
        
        a4 = a4/100
        a4 = (a4+1)**(1/12)
        a2 = a2*12
        

        # In[128]:

        #Creo il df della rendita
        lista_rendita = [a3]
        for i in range(a2+20):
            renditaX = lista_rendita[i]*a4
            lista_rendita.append(renditaX)

        lista_df_rendita = [lista_rendita[0], lista_rendita[-1]]
        df_rendite_in_fin = pd.DataFrame(lista_df_rendita, index=['Rendita mensile iniziale', 'Rendita mensile finale'], columns=['Rendita nel tempo'])
        df_rendite_in_fin = df_rendite_in_fin.transpose()

        st.write('''###  ''')
        st.write('''### Prospetto rendita indicizzata''')

        df_rendite_in_fin

        ## 
        scelta = a1
        mu = portafogli['REND.ATTESO'][scelta]
        mu = (mu+1)**(1/12)

        sigma = portafogli['''VOL.ATTESA'''][scelta]
        sigma = sigma/(12**(1/2))


        # In[151]:

        simulazioni = int(round(4900/(a2+20),0))

        def montecarlo_rendita(start, mu, sigma, rendita):
            lista_serie = []

            for i in range (simulazioni):
                lista = [start]
                lista_rend = []
                for i1 in range(a2+20):
                    rend = random.normalvariate(mu, sigma)
                    if lista[i1-1] > rendita:
                        valore = max(0,rend*(lista[i1]-lista_rendita[i1]))
                    else:
                        valore=0
        #             print("iniziale", lista[i1-1])
        #             print("rendimento",rend)
        #             print("rendita",rendita)
        #             print("finale", valore)
                    lista.append(valore)
                lista_serie.append(lista)

            df = pd.DataFrame(lista_serie)
            # df = df.transpose()

            lista_liste=[]
            for col in df.columns:
                lista_valori = [start]

            df = df.transpose()
            # df = df.cumprod()
            return df


        # In[154]:


        df = montecarlo_rendita(a0,mu, sigma, a3)
    
    if multi == True:


        st.write('''###  ''')
        st.write('''#### Prima fase''')

        a2 = st.slider('Durata primo periodo erogazione (anni)', 0,20, 10)
        a1 = st.selectbox('Seleziona il portafoglio per il primo periodo', list(portafogli.index))

        st.write('''###  ''')
        st.write('''#### Seconda fase''')         
        a22 = st.slider('Durata secondo periodo erogazione (anni)', 0,20, 10)
        a12 = st.selectbox('Seleziona il portafoglio per il secondo periodo', list(portafogli.index))       
        
        
        a4 = a4/100
        a4 = (a4+1)**(1/12)

        a2=a2*12
        a22=a22*12

        # In[128]:

        #Creo il df della rendita
        lista_rendita = [a3]
        for i in range(a2+a22+20):
            renditaX = lista_rendita[i]*a4
            lista_rendita.append(renditaX)

        lista_df_rendita = [lista_rendita[0], lista_rendita[-1]]
        df_rendite_in_fin = pd.DataFrame(lista_df_rendita, index=['Rendita mensile iniziale', 'Rendita mensile finale'], columns=['Rendita nel tempo'])
        df_rendite_in_fin = df_rendite_in_fin.transpose()

        st.write('''###  ''')
        st.write('''### Prospetto rendita indicizzata''')

        df_rendite_in_fin

        ## 
        scelta = a1
        scelta2 = a12
        mu = portafogli['REND.ATTESO'][scelta]
        mu = (mu+1)**(1/12)

        sigma = portafogli['''VOL.ATTESA'''][scelta]
        sigma = sigma/(12**(1/2))

        mu2 = portafogli['REND.ATTESO'][scelta2]
        mu2 = (mu2+1)**(1/12)

        sigma2 = portafogli['''VOL.ATTESA'''][scelta2]
        sigma2 = sigma2/(12**(1/2))


        # In[151]:

        simulazioni = int(round(4900/(a2),0))
        simulazioni2 = int(round(4900/(a22+20),0))


        def montecarlo_rendita(start, mu, sigma,mu2, sigma2, rendita):
            lista_serie = []

            for i in range (simulazioni):
                lista = [start]
                lista_rend = []

                for i1 in range(a2):
                    rend = random.normalvariate(mu, sigma)
                    if lista[i1-1] > lista_rendita[i1]:
                        valore = max(0,rend*(lista[i1]-lista_rendita[i1]))
                    else:
                        valore=0
        #             print("iniziale", lista[i1-1])
        #             print("rendimento",rend)
        #             print("rendita",rendita)
        #             print("finale", valore)
                    lista.append(valore)


                for i12 in range(i1+1, a2+a22+20):
                    rend = random.normalvariate(mu2, sigma2)
                    if lista[i12-1] > lista_rendita[i12]:
                        valore = max(0,rend*(lista[i12]-lista_rendita[i12]))
                    else:
                        valore=0
        #             print("iniziale", lista[i1-1])
        #             print("rendimento",rend)
        #             print("rendita",rendita)
        #             print("finale", valore)
                    lista.append(valore)           


                lista_serie.append(lista)

            df = pd.DataFrame(lista_serie)
            # df = df.transpose()

            lista_liste=[]
            for col in df.columns:
                lista_valori = [start]

            df = df.transpose()
            # df = df.cumprod()
            return df


        # In[154]:


        df = montecarlo_rendita(a0,mu, sigma,mu2, sigma2, a3)


    # In[156]:


    st.write('''###  ''')
    st.write('''### Rappresentazione grafica del capitale residuo (varie simulazioni)''')
    
    lista_col=[]
    lista_mese=[]
    lista_val=[]

    for col in df.columns:
        for ind in list(df.index):
            lista_col.append(str(col))
            lista_mese.append(ind)
            lista_val.append(df[col][ind])
    df_alt = pd.DataFrame(index=range(len(lista_col)))
    df_alt['Simulazione']=lista_col
    df_alt['Mese']=lista_mese
    df_alt['Capitale residuo']=lista_val
    df_alt['OT'] = a2+a22
    if multi == True:
        df_alt['OTparz'] = a2

    

    import altair as alt

    if multi == True:

        fig1 = alt.Chart(df_alt).mark_line().encode(x='Mese',y='Capitale residuo',color=alt.Color('Simulazione',legend=None),tooltip=['Capitale residuo','Mese'])
        fig2 = alt.Chart(df_alt).mark_rule(color = 'red', style='dotted').encode( x='OT',size=alt.value(4))
        fig3 = alt.Chart(df_alt).mark_rule(color = 'green', style='dotted').encode( x='OTparz',size=alt.value(4))

        immagine = fig1+fig2+fig3

    if multi == False:

        fig1 = alt.Chart(df_alt).mark_line().encode(x='Mese',y='Capitale residuo',color=alt.Color('Simulazione',legend=None),tooltip=['Capitale residuo','Mese'])
        fig2 = alt.Chart(df_alt).mark_rule(color = 'red', style='dotted').encode( x='OT',size=alt.value(4))
        

        immagine = fig1+fig2

    st.altair_chart(immagine, use_container_width=True)

    df_finale = df.head(a2+a22).tail(1)
    
    arr_finale = np.array(df_finale)
    probabilita = len(np.where(arr_finale>0)[1])/simulazioni
    probabilita = str(round(probabilita*100,2))+"%"
    
    st.write('''###  ''')
    st.write('''### Probabilità calcolate dall'algoritmo''')

    probabilita = pd.DataFrame(probabilita, index=['''Probabilità di superare l'obiettivo'''], columns=['% sulle simulazioni'])
    probabilita

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

if pagina == "Tutorials e Formazione":

    st.title('''Pagina in costruzione''')



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

if pagina == "Video Tutorial":

    st.title('''Pagina in costruzione''')  

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
# %%
