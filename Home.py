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


st.title("GeoData Visualization Platform")

#add upload data button in the sidebar for geojson files
uploaded_file = st.sidebar.file_uploader("Carga los datos de la misión que quieras procesar y/o visualizar", type=["geojson"])
if uploaded_file is not None:
    gdf_mission = gpd.read_file(uploaded_file)
    gdf_mission.rename(columns={'NO2': 'NO₂', 'C3H8': 'Propano C₃H₈', 'C4H10': 'Butano C₄H₁₀', 'CH4': 'Metano CH₄', 'H2': 'H₂', 'C2H5OH': 'Etanol C₂H₅OH'}, inplace=True)

    # Botón para procesar los datos antes de visualizarlos
    if st.sidebar.button('Procesamiento'):
        # add random noise to the data
        gdf_mission['NO₂'] = gdf_mission['NO₂'] + np.random.normal(0, 0.1, len(gdf_mission))
        st.session_state['gdf_mission'] = gdf_mission

        st.sidebar.download_button(
        label="Descargar datos",
        data=save_geojson_with_bytesio(gdf_mission),
        file_name=os.path.splitext(uploaded_file.name)[0]+'_processed.geojson',
        mime="application/geo+json",
        )  


    select_tile_provider = st.sidebar.selectbox('Select tile provider', list(tile_providers.keys()))
    select_variable = st.sidebar.selectbox('Select variable', list(gdf_mission.columns[~gdf_mission.columns.isin(['geometry','lat','lot'])]))
    df = gdf_mission[[select_variable, 'lat', 'lot','geometry']]
    st.session_state['df'] = df



if st.sidebar.button('Visualizar'):
    figs = []
    st.markdown(f"## Mapa de la misión {os.path.splitext(uploaded_file.name)[0]} para la variable {select_variable}")
    plot_points(st.session_state['df'])
    st.markdown('## Visualización de los datos')
    st.line_chart(st.session_state['df'][select_variable],use_container_width=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=st.session_state['df'].index,
        y=st.session_state['df'][select_variable],
        mode='lines+markers',
        line=dict(color='#e74c3c', width=2),
        marker=dict(symbol='circle',color='#3498db', size=5)
    ))
    figs.append(fig)
    st.markdown(f"##Histograma de la variable {select_variable}")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df[select_variable],
        marker_color='#3498db',
        opacity=0.75
    ))
    figs.append(fig)
    st.bar_chart(st.session_state['df'][select_variable],use_container_width=True)
    st.markdown(f"## {select_variable} Estadística descriptiva")
    st.table(st.session_state['df'][select_variable].describe().round(2).transpose())
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Estadístico','Valor'],
                    line_color='darkslategray',
                    align='left'),
        cells=dict(values=[list(st.session_state['df'][select_variable].describe().round(2).index),
                           list(st.session_state['df'][select_variable].describe().round(2).values)],
                     line_color='darkslategray',
                     align='left'))        
    ])
    figs.append(fig)
    # x_grid, y_grid, z_grid = create_3d_grid(df,5)
    # ax = plot_grid_and_points(df,x_grid, y_grid, z_grid, select_variable)
    # st.pyplot(ax.figure)
    st.session_state['figs'] = figs


export_as_pdf = st.button("Exportar Reporte en PDF")
if export_as_pdf:
    pdf = FPDF()
    for fig in st.session_state['figs']:
        pdf.add_page()
        with NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
            fig.write_image(tmpfile.name)
            pdf.image(tmpfile.name,10,10,200,100)
    html = create_download_link(pdf.output(dest="S").encode("latin-1"), "Reporte misión {} para la variable {}".format(os.path.splitext(uploaded_file.name)[0], select_variable))
    st.markdown(html, unsafe_allow_html=True)

st.sidebar.title('Acerca')
markdown = 'Plataforma para visualizar mediciones atmosféricas de gases contaminantes.'
st.sidebar.info(markdown)
st.sidebar.image('logo-eafit.png')