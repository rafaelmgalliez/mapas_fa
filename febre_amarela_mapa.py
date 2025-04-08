import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors
import numpy as np
from folium.plugins import HeatMap

# --- Nome do arquivo de dados com coordenadas ---
GEOCODED_CSV_PATH = "fa_casoshumanos_1994-2024_com_coords.csv"

# --- Carregar Dados Geocodificados ---
@st.cache_data
def load_geocoded_data(csv_path):
    """Carrega os dados já geocodificados do CSV."""
    try:
        df = pd.read_csv(csv_path, sep=';', encoding='utf-8', on_bad_lines='warn')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, sep=';', encoding='latin-1', on_bad_lines='warn')
        except UnicodeDecodeError:
            raise UnicodeDecodeError("Could not decode the CSV file with UTF-8 or Latin-1 encoding. Please check the file encoding.")
    except pd.errors.ParserError as e:
        st.error(f"Erro ao ler o CSV geocodificado: {e}")
        return None

    if df is not None:
        # Converter coluna de data para datetime
        df['DT_IS'] = pd.to_datetime(df['DT_IS'], format='%d/%m/%Y', errors='coerce')
        # Remover linhas com datas inválidas e coordenadas ausentes
        df.dropna(subset=['DT_IS', 'Latitude', 'Longitude', 'MUN_LPI'], inplace=True)
        # Converter Latitude e Longitude para numérico
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        df.dropna(subset=['Latitude', 'Longitude'], inplace=True)
        df['ANO_IS'] = df['DT_IS'].dt.year.astype(int) # Ensure ANO_IS is available
    return df

@st.cache_data
def aggregate_cases(df, selected_states, start_year, end_year):
    """Agrega o número de casos por município, filtrando por estado e período."""
    if df is None:
        return pd.DataFrame(columns=['MUN_LPI', 'Casos', 'Latitude', 'Longitude'])
    df_filtered = df[df['UF_LPI'].isin(selected_states)]
    df_filtered = df_filtered[(df_filtered['ANO_IS'] >= start_year) & (df_filtered['ANO_IS'] <= end_year)]
    cases_by_municipio = df_filtered.groupby(['MUN_LPI', 'Latitude', 'Longitude']).size().reset_index(name='Casos')
    cases_by_municipio = cases_by_municipio.rename(columns={'MUN_LPI': 'Municipio'})
    return cases_by_municipio

@st.cache_data
def filter_individual_cases(df, selected_states, start_year, end_year):
    """Filtra os casos individuais por estado e período."""
    if df is None:
        return pd.DataFrame(columns=['Latitude', 'Longitude', 'ANO_IS'])
    df_filtered = df[df['UF_LPI'].isin(selected_states)]
    df_filtered = df_filtered[(df_filtered['ANO_IS'] >= start_year) & (df_filtered['ANO_IS'] <= end_year)]
    return df_filtered

def create_hotspot_map(df_aggregated, min_casos, max_casos, zoom_start=4):
    """Cria um mapa de hotspots de casos por município com escala de cor e tamanho."""
    if df_aggregated is None or df_aggregated.empty:
        return None
    filtered_df = df_aggregated[(df_aggregated['Casos'] >= min_casos) & (df_aggregated['Casos'] <= max_casos)].copy()
    if filtered_df.empty:
        return None

    center_lat = filtered_df['Latitude'].mean() if not filtered_df.empty else -15.78
    center_lon = filtered_df['Longitude'].mean() if not filtered_df.empty else -47.93
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)
    cmap = LinearSegmentedColormap.from_list("yellow_red", ["yellow", "red"])
    max_filtered_casos = filtered_df['Casos'].max() if not filtered_df.empty else 1

    for index, row in filtered_df.iterrows():
        municipio = row['Municipio']
        casos = row['Casos']
        lat = row['Latitude']
        lon = row['Longitude']
        color_val = casos / max_filtered_casos if max_filtered_casos > 0 else 0
        marker_color = matplotlib.colors.to_hex(cmap(color_val))
        radius = np.sqrt(casos) * 8  # Adjust radius scale

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.6,
            tooltip=f"{municipio}: {casos} casos"
        ).add_to(m)

    if max_filtered_casos > 0:
        num_colors = 5
        color_indices = np.linspace(0, 1, num_colors)
        legend_html_hotspot = f"""
             <div style="position: fixed;
                         bottom: 50px; left: 50px; width: 170px; height: auto;
                         border:2px solid grey; z-index:9999; font-size:14px; background-color:white; opacity:0.9;">
               <b style="text-align: center;">Casos de Febre Amarela</b><br>
             """
        for i in range(num_colors):
            color = matplotlib.colors.to_hex(cmap(color_indices[i]))
            value = int(max_filtered_casos * color_indices[i])
            legend_html_hotspot += f'<i style="background:{color}; opacity:1; border-radius:50%; width: 10px; height: 10px; display: inline-block; margin-right: 5px;"></i> {value}+ casos<br>'
        legend_html_hotspot += """
             </div>
             """
        m.get_root().html.add_child(folium.Element(legend_html_hotspot))

    return m

def create_kde_heatmap(df, zoom_start=4, blur=20, radius=25):
    """Cria um mapa de heatmap de densidade de kernel dos casos individuais."""
    if df is None or df.empty:
        return None

    locations = df[['Latitude', 'Longitude']].values.tolist()

    if not locations:
        return None

    center_lat = np.mean([loc[0] for loc in locations]) if locations else -15.78
    center_lon = np.mean([loc[1] for loc in locations]) if locations else -47.93
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)
    HeatMap(locations, blur=blur, radius=radius).add_to(m)
    return m

# --- STREAMLIT APP ---
st.title("Mapa de Casos de Febre Amarela (Geocodificados)")

df_geocoded = load_geocoded_data(GEOCODED_CSV_PATH)

if df_geocoded is not None:
    all_states = sorted(df_geocoded['UF_LPI'].unique())
    all_states_with_all = ["Todos"] + all_states
    default_states = ['PA'] if 'PA' in all_states else ["Todos"] if "Todos" in all_states_with_all else [all_states[0]] if all_states else []
    selected_states = st.multiselect("Selecionar Estados:", all_states_with_all, default=default_states)

    if "Todos" in selected_states:
        selected_states_for_filter = all_states
    else:
        selected_states_for_filter = selected_states

    min_year = int(df_geocoded['ANO_IS'].min()) if not df_geocoded.empty else 1994
    max_year = int(df_geocoded['ANO_IS'].max()) if not df_geocoded.empty else 2024

    year_range = st.slider("Selecionar Período:", min_value=min_year, max_value=max_year, value=(min_year, max_year))
    start_year_selected, end_year_selected = year_range

    df_aggregated = aggregate_cases(df_geocoded, selected_states_for_filter, start_year_selected, end_year_selected)

    if not df_aggregated.empty:
        max_casos_municipio = int(df_aggregated['Casos'].max()) if not df_aggregated.empty else 1
        casos_range_municipio = st.slider(
            "Filtrar municípios por número de casos:",
            min_value=1,
            max_value=max_casos_municipio,
            value=(1, max_casos_municipio)
        )
        min_casos_selected_municipio, max_casos_selected_municipio = casos_range_municipio

        st.subheader("Mapa de Hotspots de Casos de Febre Amarela por Município")
        hotspot_map = create_hotspot_map(
            df_aggregated.copy(),
            min_casos_selected_municipio,
            max_casos_selected_municipio
        )
        if hotspot_map:
            st_folium(hotspot_map, width=800, height=600)
        else:
            st.warning("Nenhum município encontrado com os filtros selecionados para o mapa de hotspots.")

    st.subheader("Mapa de Densidade de Kernel de Casos Individuais")
    df_filtered_kde = filter_individual_cases(df_geocoded, selected_states_for_filter, start_year_selected, end_year_selected)
    if not df_filtered_kde.empty:
        kde_map = create_kde_heatmap(df_filtered_kde)
        if kde_map:
            st_folium(kde_map, width=800, height=600)
        else:
            st.warning("Não foi possível gerar o mapa de densidade de kernel com os filtros selecionados.")
    else:
        st.warning("Não há dados para gerar o mapa de densidade de kernel com os filtros selecionados.")

else:
    st.error("Falha ao carregar os dados geocodificados.")
