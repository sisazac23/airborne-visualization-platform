import pandas as pd
import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
import folium 
import numpy as np
from fpdf import FPDF
import base64
from tempfile import NamedTemporaryFile
from streamlit_folium import folium_static
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from hampel import hampel
import scipy as sp
import warnings 
import pickle
from sklearn.preprocessing import StandardScaler
from math import sqrt
from shapely.geometry import Point, Polygon, LineString, MultiLineString
from data_processing import load_data_txt, load_data, classify_route, get_denoised_signal, create_binary_df, df_to_geojson_anomalies, signal_anomaly_neighborhood, df_to_geojson_neighborhood, predict_signal_h2, predict_signal_ch4, plot_geojson, path_plot_3d
from PIL import Image

warnings.filterwarnings("ignore")


import glob
import os
import io




st.set_page_config(layout="wide")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

tile_providers = {
    'OpenStreetMap': 'OpenStreetMap',
    'Stamen Terrain': 'Stamen Terrain',
    'Stamen Toner': 'Stamen Toner',
    'CartoDB Positron': 'CartoDB Positron',
    'CartoDB Dark_Matter': 'CartoDB Dark_Matter',
}


# Funciones

def plot_over_map(gdf_anomalies,df_1, gdf_neighborhood, select_variable):
    stream_map = folium.Map(location=[6.2518400, -75.5635900], zoom_start=10, control_scale=True, tiles=select_tile_provider,locate_control=True, latlon_control=True, draw_export=True, minimap_control=True)
    var_anomaly = select_variable + '_anomaly'
    outliers = np.where(gdf_anomalies[var_anomaly] == 1)[0]  
    flight_trajectory = np.where(gdf_anomalies[var_anomaly] == 0)[0]         
    for i in outliers:
        location = [gdf_anomalies.iloc[i].geometry.y, gdf_anomalies.iloc[i].geometry.x]
        folium.CircleMarker(location=location, radius=12, color='red', fill=True, fill_color='red').add_to(stream_map)
            
    for i in flight_trajectory:
        location = [gdf_anomalies.iloc[i].geometry.y, gdf_anomalies.iloc[i].geometry.x]
        folium.CircleMarker(location=location, radius=5, color='blue', fill=True, fill_color='blue').add_to(stream_map) 
    folium.LayerControl().add_to(stream_map)
    folium_static(stream_map)

def save_geojson_with_bytesio(dataframe):
    #Function to return bytesIO of the geojson
    shp = io.BytesIO()
    dataframe.to_file(shp,  driver='GeoJSON')
    return shp

def create_download_link(val, filename):
    b64 = base64.b64encode(val)  
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'

def plot_points(data, size=10000, color='red'):
    stream_map = folium.Map(location=[6.2518400, -75.5635900], zoom_start=10, control_scale=True, tiles=select_tile_provider,locate_control=True, latlon_control=True, draw_export=True, minimap_control=True)
    for i in range(len(data)):
        location = [data.iloc[i].geometry.y, data.iloc[i].geometry.x]
        folium.CircleMarker(location=location, radius=5, color=color, fill=True, fill_color=color).add_to(stream_map)

    folium.LayerControl().add_to(stream_map)
    
    folium_static(stream_map)




#add upload data button in the sidebar for geojson files
uploaded_file = st.sidebar.file_uploader("Carga los datos de la misión que quieras procesar y/o visualizar", type=["txt","csv"])

st.title("Plataforma de Visualización de Gases Atmosféricos")



select_tile_provider = st.sidebar.selectbox('Select tile provider', list(tile_providers.keys()))
gases=['CO', 'NO₂', 'Propano C₃H₈', 'Butano C₄H₁₀', 'Metano CH₄', 'H₂', 'Etanol C₂H₅OH']
select_variable = st.sidebar.selectbox('Select variable', gases)


import streamlit as st
import pandas as pd
import os



if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.txt') or uploaded_file.name.endswith('.TXT'):
            df_1 = load_data_txt(uploaded_file)
            st.session_state['df_raw'] = df_1 
        else: 
            df_1 = pd.read_csv(uploaded_file)
            df_1.rename(columns={'NO2': 'NO₂', 'C3H8': 'Propano C₃H₈', 'C4H10': 'Butano C₄H₁₀', 'CH4': 'Metano CH₄', 'H2': 'H₂', 'C2H5OH': 'Etanol C₂H₅OH'}, inplace=True)
    except Exception as e:
        st.error(f"Asegura que sean datos del sensor. Error al cargar los datos del archivo. Error: {e}")

    if st.sidebar.button('Procesamiento'):
        try:
            fecha_mision = df_1['Date'][0]
            date = fecha_mision.split('-')
            fecha_mision = date[2] + '/' + date[1] + '/20' + date[0]
            st.header(f'Fecha de la misión: {fecha_mision}')
            df_denoised = get_denoised_signal(df_1, 0.1, gases)
            df_1.loc[:, gases] = df_denoised[gases].values
            st.subheader('Estadísticas descriptivas')
            st.write(df_1.describe())
            df_1 = create_binary_df(df_1, 20, gases)
            label = classify_route(df_1)
            st.session_state['df_1'] = df_1
            st.session_state['label'] = label
        except Exception as e:
            st.error(f"Error durante el procesamiento de los datos. Error: {e}")

        try:
            gpd_anomal = df_to_geojson_anomalies(df_1, df_1.columns)
            st.session_state['gpd_anomal'] = gpd_anomal

            signal_neigh = signal_anomaly_neighborhood(df_1, gases, 3)
            gdp_neigh = df_to_geojson_neighborhood(signal_neigh, gases)
            st.session_state['gdp_neigh'] = gdp_neigh

            gpd_merged = gpd_anomal.merge(gdp_neigh, on=['lat', 'lot', 'geometry'], how='left')
        except Exception as e:
            st.error(f"Error al crear GeoJSONs o al unir DataFrames: {e}")

        try:
            # Asegúrate de que la función 'save_geojson_with_bytesio' esté correctamente definida
            geojson_bytes = save_geojson_with_bytesio(gpd_anomal)
            file_name = os.path.splitext(uploaded_file.name)[0] + '_processed.geojson'
            st.sidebar.download_button(
                label="Descargar datos",
                data=geojson_bytes,
                file_name=file_name,
                mime="application/geo+json",
            )
        except Exception as e:
            st.error(f"Error al preparar el archivo para la descarga. Error: {e}")

    
       
if st.sidebar.button('Visualizar'):
    figs = []
    figs_all_variables = []
    plot_over_map(st.session_state['gpd_anomal'],st.session_state['df_1'],st.session_state['gdp_neigh'], select_variable)
    if st.session_state['label'] == 1:
        st.header('Ruta realizada por un helicóptero')
        st.image('figures/hipae.png')
    else:
        st.header('Ruta realizada por un aeroplano')
        st.image('figures/caravan.png')
    _, fig = predict_signal_h2(st.session_state['df_1'],'Pipeline/models/linear_reg_mult_input_H2.sav',['Metano CH₄','CO','A_1','Propano C₃H₈','Butano C₄H₁₀','T_3'])
    figs.append(fig)
    st.pyplot(fig)
   
    _ , fig =  predict_signal_ch4(st.session_state['df_1'],'Pipeline/models/linear_reg_mult_input_CH4.sav',['A_1', 'CO', 'Propano C₃H₈', 'T_3', 'Etanol C₂H₅OH', 'Butano C₄H₁₀',])
    figs.append(fig)
    st.pyplot(fig)


    plots_dict = plot_geojson(st.session_state['gpd_anomal'],st.session_state['gdp_neigh'],gases)
    for gas in gases:
        # gráfica de anomalías
        fig1 = plots_dict[gas+'_anomalies']
        figs.append(fig1)
        fig1.savefig("gas_anomaly.png")

        # gráfica de vecindarios
        fig2 = plots_dict[gas+'_neighborhoods']
        figs.append(fig2)
        fig2.savefig("gas_neighborhood.png")

        # gráfica 3d
        fig3 = path_plot_3d(st.session_state['df_1'],gas)
        figs.append(fig3)
        fig3.savefig("gas_3d.png")
        
        image1 = Image.open("gas_anomaly.png")
        st.image(image1)
        image2 = Image.open("gas_neighborhood.png")
        #st.image(image2)
        image3 = Image.open("gas_3d.png")
        #st.image(image3)

        os.remove("gas_anomaly.png")
        os.remove("gas_neighborhood.png")
        os.remove("gas_3d.png")

    for column in st.session_state['df_raw'].columns[2:]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=st.session_state['df_raw']['time'], y=st.session_state['df_raw'][column], mode='lines', name=column))
        fig.update_layout(title='{}'.format(column), xaxis_title='Tiempo', yaxis_title='{}'.format(column))
        figs_all_variables.append(fig)

    st.session_state['figs'] = figs
    st.session_state['figs_all_variables'] = figs_all_variables
 

export_as_pdf = st.button("Exportar Reporte en PDF")
if export_as_pdf:
    try:
        st.write('Exportando reporte en PDF')
        pdf = FPDF()
        
        for fig in st.session_state['figs']:
            pdf.add_page()
            with NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                fig.savefig(tmpfile.name)
                pdf.image(tmpfile.name, 10, 10, 200, 100)
        
        for fig in st.session_state['figs_all_variables']:
            pdf.add_page()
            with NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                fig.write_image(tmpfile.name)
                pdf.image(tmpfile.name, 10, 10, 200, 100)
        
        # Assuming create_download_link is a function defined elsewhere
        html = create_download_link(
            pdf.output(dest="S").encode("latin-1"),
            "reporte_mision_{}".format(
                os.path.splitext(uploaded_file.name)[0], select_variable
            )
        )
        st.markdown(html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f'Visualiza primero las mediciones. Estas son las que se generan en el reporte. Error: {e}')

st.sidebar.title('Acerca')
markdown = 'Plataforma para visualizar mediciones atmosféricas de gases contaminantes.'
st.sidebar.info(markdown)
st.sidebar.image('figures/logo-eafit.png')
