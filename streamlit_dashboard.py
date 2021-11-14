import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt 
import time
import seaborn as sns
from PIL import Image 
import webbrowser
import time
import pickle
import json
import altair as alt


st.set_page_config(page_title='My ElysE App', page_icon=':smiley', 
                   layout="wide", initial_sidebar_state='expanded')

# Paths of datasets
path0 = "df_principal.csv"
path1 = "df_principal_cleaned.csv"

def Decorator (function) :
    def modified_function(df):
        time_ = time.time()
        res = function(df)
        time_ = time.time()-time_
        with open(f"{function.__name__}_exec_time.txt","w") as f:
            f.write(f"{time_}")
        return res
    return modified_function



# import data
@Decorator
@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def import_data (path) :
    df = pd.read_csv (path).sample(frac=0.5)
    #remove default csv index
    df.drop(["Unnamed: 0"], axis = 1 , inplace = True)
    return df

# parsing date
@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def parse_date (df):
    # Convert date_mutation dtypes (object => datetime)
    df ["date_mutation"] = df ["date_mutation"].map(pd.to_datetime)   
    return df 


# Delete rows with NaN & and return 10% of cleaned data
@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def delete_nan (df) :
    df_ = df.copy()
    # indice des lignes dont les valeurs foncières sont vides 
    index_nan = df.index[df[df.columns].isnull().any(axis = 1)]
    #index_nan.shape
    # supprimer ces lignes
    df_.drop(index_nan , 0, inplace = True)
    # réinitialiser les indices
    df_.reset_index(drop = True, inplace=True)
    df_ = df_.sample(frac=0.10)
    return df_

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def count_rows (x):
    return len(x)

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def coord_geo (df) :
    return df[["latitude", "longitude"]]


@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def hist (x) :
    fig, ax = plt.subplots(figsize=(2,2))
    ax.tick_params(axis='x', rotation=90)
    ax.hist(x)
    return fig

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def pie (x) :
    fig, ax = plt.subplots(figsize=(2,2))
    ax.pie(x)
    return fig

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def scatter (x,y) :
    fig, ax = plt.subplots(figsize=(2,2))
    #ax.tick_params(axis='x', rotation=90)
    ax.scatter(x,y)
    return fig

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def hist_year (x) :
    n_bins =20
    fig, (ax1,ax2) = plt.subplots(1,2, tight_layout = True)
    ax1.hist(x["nature_mutation"] , bins = n_bins)
    ax1.set_title ("Nature Mutation")
    ax1.tick_params(axis='x', rotation=30)
    
    ax2.hist(x["type_local"] , bins = n_bins)
    ax2.set_title ("type_local")
    ax2.tick_params(axis='x', rotation=30)

    
    return fig
    


@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def group_month_year(df):    
    def count_rows(rows):
        return len(rows)
    gr_by = df.groupby(["month","year"]).apply(count_rows).unstack()
    return gr_by

st.cache(allow_output_mutation=True,suppress_st_warning=True)
def group_month_day(df):    
    def count_rows(rows):
        return len(rows)
    gr_by = df.groupby(["month","day"]).apply(count_rows).unstack()
    return gr_by


@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def heatmap(unstacked):
    fig, ax= plt.subplots(figsize=(3,3))
    ax = sns.heatmap(unstacked, linewidths=0.5)
    return fig




def main () : 
    
    
    #st.set_option('deprecation.showPyplotGlobalUse', False)
    
    # import data preprocessed and cleaned
    df_cleaned = import_data(path1)
    
    # extract cleaned data
    #df_cleaned = delete_nan(df)
    #df_cleaned = parse_date(df_cleaned)
    
    # extract data by date
    df_2016 = df_cleaned.loc[(df_cleaned["year"]== 2016) , df_cleaned.columns].reset_index(drop=True)
    df_2017 = df_cleaned.loc[(df_cleaned["year"]== 2017) , df_cleaned.columns].reset_index(drop=True)
    df_2018 = df_cleaned.loc[(df_cleaned["year"]== 2018) , df_cleaned.columns].reset_index(drop=True)
    df_2019 = df_cleaned.loc[(df_cleaned["year"]== 2019) , df_cleaned.columns].reset_index(drop=True)
    df_2020 = df_cleaned.loc[(df_cleaned["year"]== 2020) , df_cleaned.columns].reset_index(drop=True)
    
    # columns of df cleaned
    col_df = df_cleaned.columns
    
    
    
    st.sidebar.header ("WELCOME")
    

    # Exploratory Data Analysis
    menuEDA = [ "Choose EDA", "General", "EDA 2020" ,"EDA 2019", "EDA 2018", "EDA 2017", "EDA 2016"]
    EDA = st.sidebar.selectbox ("Which year do you want to analyse?:", menuEDA)

    # Menu principal
    menu = ["", "About" , "Prediction", "Contact me"]
    choices = st.sidebar.selectbox ("Informations:", menu)
    
        
    if EDA == "Choose EDA":
         st.markdown("<h1 style='text-align: center; color: blue;'>Rasel Immo</h1>", unsafe_allow_html=True) # Ras : Rasoloarivony # El : Elysé
         st.markdown("<h5 style='text-align: center; color: green;'>Have a good navigation</h5>", unsafe_allow_html=True)
         #st.text("Let's discover together")
         img = Image.open ("immobilier.jpg")
         st.image (img, use_column_width = True)

    
    if EDA == "General": 
        st.write("Here you will find the original data dashboard from 2016 to 2020")
        
        my_expander1 = st.expander(label='Dataset')
        with my_expander1:
            'from 2016 to 2020'
        
            radioG1 = st.radio(
                "Dataset",
                ("head", "describe", "full") )
            if  radioG1 == "head":
                    st.write( df_cleaned.head() )
            elif radioG1== "describe":
                    st.write( df_cleaned.describe() )
            elif radioG1 == "full":
                    cols = st.multiselect(label="choose columns that you want to display", options=col_df)
                    if cols :
                        st.write(df_cleaned[cols])
                    select_all = st.button("ALL")
                    if select_all :
                        df_cleaned

        my_expander2 = st.expander(label='Figures')
        with my_expander2:
            ''
        
            radioG1 = st.radio(
                "",
                ("correlation", "mutation", "type local", "map") )
            if radioG1 == "correlation":
                    st.pyplot(heatmap(df_cleaned.corr()) ) 
            elif radioG1== "mutation":
                ca , cb = st.columns(2)
                ca.pyplot( heatmap(group_month_year(df_cleaned)) )

                by_year = df_cleaned.groupby(by=["year"]).apply(count_rows)
                cb.line_chart(by_year)

            elif radioG1 == "type local":
                c1,c2= st.columns(2)
                with c1 :
                    if st.checkbox("histogram"):
                        st.pyplot(hist(df_cleaned["type_local"]))

                with c2 :    
                    if st.checkbox("lines"):                   
                        st.pyplot(hist(df_cleaned.groupby("type_local").year.value_counts().unstack(0)) )
            
            elif radioG1 == "map":
                index = st.selectbox ("choose one", df_cleaned.valeur_fonciere.index)
                if index : 
                    st.map(coord_geo(df_cleaned[index]))
                



            

    
    
    if EDA == "EDA 2020" : 
        st.header ("DATA 2020")

        if st.checkbox ("dataset"):  
            df_2020
            st.markdown("**Describe**")
            st.write (df_2020.describe())

            st.markdown("**Check this values**")
            year=2020
            
            st.write (f"mean of land value {year}  : {df_2020.valeur_fonciere.mean()} euros")
            st.write (f"maximum of land value {year}  : {df_2020.valeur_fonciere.max()} euros")
            st.write (f"minimum of land value {year}  : {df_2020.valeur_fonciere.min()} euros")

        
        if st.checkbox ("display local type"):

            c = alt.Chart(df_2020).mark_circle().encode(
                    x='surface_terrain', y='valeur_fonciere')

            st.altair_chart(c, use_container_width=True)
            
            
        if st.checkbox ("map"):
            st.map(coord_geo(df_2020))

        if st.checkbox ("mutation"):
            st.pyplot(heatmap(group_month_day(df_2020)))

        if st.checkbox ("nature mutation and type local"):
            st.pyplot(hist_year(df_2020))

        if st.checkbox("mutation by months"):
            by_months = df_2020.groupby(by=["month"]).apply(count_rows)
            st.line_chart(by_months)

    if EDA == "EDA 2019" :
        st.header("DATA 2019")
        if st.checkbox ("dataset"):
            df_2019
            st.markdown("**Describe**")
            st.write (df_2019.describe())

            year=2019
            st.markdown("**Check this values**")

            st.write (f"mean of land value {year}  : {df_2019.valeur_fonciere.mean()} euros")
            st.write (f"maximum of land value {year}  : {df_2019.valeur_fonciere.max()} euros")
            st.write (f"minimum of land value {year}  : {df_2019.valeur_fonciere.min()} euros")

        
        if st.checkbox ("surface_terrain vs valeur_fonciere"):
            c = alt.Chart(df_2019).mark_circle().encode(
                    x='surface_terrain', y='valeur_fonciere')

            st.altair_chart(c, use_container_width=True)

        if st.checkbox ("map"):
            st.map(coord_geo(df_2019))
        
        if st.checkbox ("mutation"):
            st.pyplot(heatmap(group_month_day(df_2019)))

        if st.checkbox ("nature mutation and type local"):
            st.pyplot(hist_year(df_2019))

        if st.checkbox("mutation by months"):
            by_months = df_2019.groupby(by=["month"]).apply(count_rows)
            st.line_chart(by_months)


    if EDA =="EDA 2018" :
        st.header("DATA 2018")
        if st.checkbox ("dataset"):
            
            st.markdown("**Describe**")

            st.write (df_2018.describe())
            
            year=2018
            st.markdown("**Check this values**")

            st.write (f"mean of land value {year}  : {df_2018.valeur_fonciere.mean()} euros")
            st.write (f"maximum of land value {year}  : {df_2018.valeur_fonciere.max()} euros")
            st.write (f"minimum of land value {year}  : {df_2018.valeur_fonciere.min()} euros")


        if st.checkbox ("map"):
            st.map(coord_geo(df_2018))
        
        if st.checkbox ("surface_terrain vs valeur_fonciere"):
            c = alt.Chart(df_2018).mark_circle().encode(
                    x='surface_terrain', y='valeur_fonciere')

            st.altair_chart(c, use_container_width=True)
        
        if st.checkbox ("mutation"):
            st.pyplot(heatmap(group_month_day(df_2018)))
        
        if st.checkbox ("nature mutation and type local"):
            st.pyplot(hist_year(df_2018))

        if st.checkbox("mutation by months"):
            by_months = df_2018.groupby(by=["month"]).apply(count_rows)
            st.line_chart(by_months)


    if EDA == "EDA 2017" :
        st.header("DATA 2017")
        if st.checkbox ("dataset 2020"):
            df_2017
            st.markdown("**Describe**")
            st.write (df_2017.describe())

            year=2017
            st.markdown("**Check this values**")

            st.write (f"mean of land value {year}  : {df_2017.valeur_fonciere.mean()} euros")
            st.write (f"maximum of land value {year}  : {df_2017.valeur_fonciere.max()} euros")
            st.write (f"minimum of land value {year}  : {df_2017.valeur_fonciere.min()} euros")

        
        if st.checkbox ("surface_terrain vs valeur_fonciere"):
            c = alt.Chart(df_2017).mark_circle().encode(
                    x='surface_terrain', y='valeur_fonciere')

            st.altair_chart(c, use_container_width=True)
        if st.checkbox ("map"):
            st.map(coord_geo(df_2017))
        
        if st.checkbox ("mutation"):
            st.pyplot(heatmap(group_month_day(df_2017)))

        if st.checkbox ("nature mutation and type local"):
            st.pyplot(hist_year(df_2017))

        if st.checkbox("mutation by months"):
            by_months = df_2017.groupby(by=["month"]).apply(count_rows)
            st.line_chart(by_months)

    if EDA == "EDA 2016" :
        st.header("DATA 2016")
        if st.checkbox ("dataset"):
            df_2016
            st.write (df_2016.describe())
            
            year=2016
            st.markdown("**Check this values**")

            st.write (f"mean of land value {year}  : {df_2016.valeur_fonciere.mean()} euros")
            st.write (f"maximum of land value {year}  : {df_2016.valeur_fonciere.max()} euros")
            st.write (f"minimum of land value {year}  : {df_2016.valeur_fonciere.min()} euros")

        
        if st.checkbox ("surface_terrain vs valeur_fonciere"):
            c = alt.Chart(df_2016).mark_circle().encode(
                    x='surface_terrain', y='valeur_fonciere')

            st.altair_chart(c, use_container_width=True)

        if st.checkbox ("map"):
            st.map(coord_geo(df_2016))
        
        if st.checkbox ("mutation"):
            st.pyplot(heatmap(group_month_day(df_2016)))
        
        if st.checkbox ("nature mutation and type local"):
            st.pyplot(hist_year(df_2016))

        if st.checkbox("mutation by months"):
            by_months = df_2016.groupby(by=["month"]).apply(count_rows)
            st.line_chart(by_months)

        



            


    if choices =="":
        st.write("")
    
    
    if choices == "About" :
        st.subheader ("About")
        st.write ("With this app, you can explore all sales from 2016 to 2020 , In the general part, you will see as whole ")
        st.write (" You can also predict your land value according to the caracteristics which you want")
        if st.button("Here are the sources of datasets"):
                url="https://drive.google.com/drive/folders/1R_9A9yPOzRQzMCyTDBEJms0u1ZCN7MbY"
                webbrowser.open_new_tab(url) 


    if choices == "Prediction":

        # Here I used an unwell fitted model, just to see how it works! 
        
        st.subheader ("Predict your Land Value")
        filename = "finalized_model.sav"
        loaded_model = pickle.load(open( filename, 'rb'))

        predict_with_number = st.expander(label='Enter caracteristics')
        with predict_with_number:

            min_long = df_cleaned["longitude"].min()
            max_long = df_cleaned["longitude"].max()
            min_lat = df_cleaned["latitude"].min()
            max_lat = df_cleaned["latitude"].max()

            code_postal = st.number_input("Postal Code", min_value=10000, max_value=99000, value=75000, step=100)
            surface_terrain = st.number_input ('Area', min_value=1.4, max_value=10000.4, value=50.1, step=50.1, format="%.2f")
            longitude = st.slider (label = 'longitude', min_value = min_long, max_value = max_long, value=5.1, step=0.2,format="%.2f")
            latitude = st.slider( label = 'latitude', min_value = min_lat, max_value = max_lat, value=45.1, step=1.1)
        
            xval = np.asarray([code_postal,surface_terrain,longitude,latitude]).T.reshape(1,-1)   #list => array & => transpose 
            
            result_val = loaded_model.predict (xval)
            with st.spinner("Waiting .."):
	            time.sleep(2)
            st.success("Congratulation!")
            st.write (f"your future land value costs : {abs(result_val[0])} euros :) ")
            st.warning("This model is not reliable, It was just a test in order to seeing all streamlit functionnalities. Check in Model.py all models I tested")
            
            
            my_bar = st.progress(0)
            
            for percent_complete in range(100):
                time.sleep(0.1)
                my_bar.progress(percent_complete + 1)
            
            obj = {"Postal code ": code_postal, "Area (m2)": surface_terrain, "longitude": longitude, "latitude": latitude, "predicted price": abs(result_val[0])  }
            st.write(obj)
            st.markdown("<h5 style='text-align: center; color: yellow;'>Check the place ;) </h5>", unsafe_allow_html=True)

            good = pd.DataFrame( {'latitude':latitude, 'longitude': longitude}, index=[0] )
            st.map(good)
            st.balloons()




        
    if choices == "Contact me" :
        st.markdown ("Please, enter your name and your remark of this app")
        
        info_utils = st.text_input("Enter your Name", max_chars = 50)
        if info_utils :
            st.write (f"Hello {info_utils}")
        comment = st.text_area ("Do you have 5 min to comment?")
        
        if comment:
            st.write ("Thank you for you participation :)")
        date = st.date_input("Date")
        st.write ("Thank you")
        
        
        my_expander = st.expander(label='Contact me here')
        with my_expander: 
            'Hello there!'
            if st.button("LinkedIn"):
                urll="https://www.linkedin.com/in/elys%C3%A9-rasoloarivony-078a301b1/"
                webbrowser.open_new_tab(urll) 
            
            if st.button('Mail adress'):
                st.info ("relysmiadantsoa@yahoo.fr or \n elyserasoloarivony@gmail.com")

        st.write("I hope this app data explorer tool is useful for you!")





    

        
    

if __name__ == "__main__" :    
    main()
