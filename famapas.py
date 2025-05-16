import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from folium.plugins import HeatMap

# --- Configurações ---
GEOCODED_CSV_PATH = "fa_casoshumanos_1994-2024_com_coords.csv"
DEFAULT_ZOOM = 4
DEFAULT_CENTER = [-15.78, -47.93]  # Centro do Brasil
CMAP_COLORS = ["yellow", "red"]

# --- Funções de Carregamento e Processamento de Dados ---
@st.cache_data(show_spinner="Carregando dados...")
def load_geocoded_data(csv_path):
    """Carrega e processa dados geocodificados do CSV."""
    encodings = ['utf-8', 'latin-1']
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, sep=';', encoding=encoding, on_bad_lines='warn')
            break
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            if encoding == encodings[-1]:
                st.error(f"Erro ao ler o CSV: {e}. Verifique a codificação ou formato do arquivo.")
                return None
    else:
        return None

    try:
        df['DT_IS'] = pd.to_datetime(df['DT_IS'], format='%d/%m/%Y', errors='coerce')
        df = df.dropna(subset=['DT_IS', 'Latitude', 'Longitude', 'MUN_LPI', 'UF_LPI'])
        df = df.assign(
            Latitude=pd.to_numeric(df['Latitude'], errors='coerce'),
            Longitude=pd.to_numeric(df['Longitude'], errors='coerce'),
            IDADE=pd.to_numeric(df['IDADE'], errors='coerce'),
            ANO_IS=df['DT_IS'].dt.year.astype('Int64'),
            OBITO=df['OBITO'].fillna('NÃO').str.upper()
        ).dropna(subset=['Latitude', 'Longitude'])
        return df
    except Exception as e:
        st.error(f"Erro ao processar dados: {e}")
        return None

@st.cache_data
def aggregate_cases(df, selected_states, start_year, end_year):
    """Agrega casos por município, filtrando por estado e período."""
    if df is None or df.empty:
        return pd.DataFrame(columns=['MUN_LPI', 'Casos', 'Latitude', 'Longitude']), pd.DataFrame()
    df_filtered = df[
        (df['UF_LPI'].isin(selected_states)) &
        (df['ANO_IS'].ge(start_year)) & (df['ANO_IS'].le(end_year))
    ]
    df_aggregated = (
        df_filtered.groupby(['MUN_LPI', 'Latitude', 'Longitude'])
        .size()
        .reset_index(name='Casos')
        .rename(columns={'MUN_LPI': 'Municipio'})
    )
    return df_aggregated, df_filtered

@st.cache_data
def filter_individual_cases(df, selected_states, start_year, end_year):
    """Filtra casos individuais por estado e período."""
    if df is None or df.empty:
        return pd.DataFrame(columns=['Latitude', 'Longitude', 'ANO_IS'])
    return df[
        (df['UF_LPI'].isin(selected_states)) &
        (df['ANO_IS'].ge(start_year)) & (df['ANO_IS'].le(end_year))
    ][['Latitude', 'Longitude', 'ANO_IS']]

# --- Funções de Visualização ---
def create_hotspot_map(df_aggregated, min_casos, max_casos, df_filtered, zoom_start=DEFAULT_ZOOM):
    """Cria mapa de hotspots com escala de cor e tamanho."""
    if df_aggregated is None or df_aggregated.empty:
        return None
    filtered_df = df_aggregated[
        (df_aggregated['Casos'].ge(min_casos)) & (df_aggregated['Casos'].le(max_casos))
    ].copy()
    if filtered_df.empty:
        return None

    center = [
        filtered_df['Latitude'].mean() or DEFAULT_CENTER[0],
        filtered_df['Longitude'].mean() or DEFAULT_CENTER[1]
    ]
    m = folium.Map(location=center, zoom_start=zoom_start)

    cmap = plt.cm.get_cmap('YlOrRd')
    max_casos = filtered_df['Casos'].max() or 1

    for _, row in filtered_df.iterrows():
        color_val = row['Casos'] / max_casos
        color = mcolors.to_hex(cmap(color_val))
        radius = np.sqrt(row['Casos']) * 8
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            tooltip=f"{row['Municipio']}: {row['Casos']} casos"
        ).add_to(m)

    if max_casos > 0:
        # Legenda principal: reduzida em 50% e com fundo transparente (75% alpha)
        legend_html = """
        <div style="position: fixed; bottom: 25px; left: 25px; width: 85px; height: auto;
                    border:2px solid grey; z-index:9999; font-size:7px; background-color:rgba(255,255,255,0.75); color:black;">
            <b style="text-align: center; display: block; color:black;">Casos de Febre Amarela</b>
            {}
        </div>
        """
        legend_items = ''.join(
            f'<i style="background:{mcolors.to_hex(cmap(i))}; opacity:1; border-radius:50%; width: 5px; height: 5px; display: inline-block; margin-right: 2.5px;"></i> {int(max_casos * i)}+ casos<br>'
            for i in np.linspace(0, 1, 5)
        )
        m.get_root().html.add_child(folium.Element(legend_html.format(legend_items)))

        # Legenda de estatísticas
        total_casos = int(filtered_df['Casos'].sum())
        total_municipios = len(df_filtered['MUN_LPI'].unique())
        total_obitos = int(df_filtered['OBITO'].eq('SIM').sum())
        media_idade = round(df_filtered['IDADE'].mean(), 1) if not df_filtered['IDADE'].isna().all() else 'N/D'
        letalidade = round((total_obitos / total_casos * 100), 2) if total_casos > 0 else 0

        stats_html = """
        <div style="position: fixed; top: 25px; right: 25px; width: 160px; height: auto;
                    border:2px solid grey; z-index:9999; font-size:11px; background-color:rgba(255,255,255,0.75); color:black; text-align:left; padding: 5px;">
            <b>Estatísticas do Período</b><br>
            Total de Casos: {:,}<br>
            Municípios com Casos: {:,}<br>
            Total de Óbitos: {:,}<br>
            Média de Idade: {} anos<br>
            Letalidade: {}%
        </div>
        """
        m.get_root().html.add_child(folium.Element(stats_html.format(total_casos, total_municipios, total_obitos, media_idade, letalidade)))

    return m

def create_kde_heatmap(df, zoom_start=DEFAULT_ZOOM, blur=20, radius=25, gradient=None):
    """Cria mapa de densidade de kernel com parâmetros ajustáveis."""
    if df is None or df.empty:
        return None
    locations = df[['Latitude', 'Longitude']].values.tolist()
    if not locations:
        return None

    center = [
        np.mean([loc[0] for loc in locations]) or DEFAULT_CENTER[0],
        np.mean([loc[1] for loc in locations]) or DEFAULT_CENTER[1]
    ]
    m = folium.Map(location=center, zoom_start=zoom_start)
    HeatMap(locations, blur=blur, radius=radius, gradient=gradient).add_to(m)
    return m

# --- Função Auxiliar para Seleção de Estados ---
def get_selected_states(all_states, selected):
    """Retorna lista de estados selecionados, tratando 'Todos'."""
    return all_states if 'Todos' in selected else selected

# --- Streamlit App ---
st.set_page_config(page_title="Mapa de Casos de Febre Amarela", layout="wide")
st.title("Mapa de Casos de Febre Amarela (Geocodificados)")

# Carregar dados
df_geocoded = load_geocoded_data(GEOCODED_CSV_PATH)

if df_geocoded is not None:
    all_states = sorted(df_geocoded['UF_LPI'].unique())
    all_states_with_all = ["Todos"] + all_states
    default_states = ["Todos"]  # Padrão: apenas "Todos" selecionado
    min_year = int(df_geocoded['ANO_IS'].min() or 1994)
    max_year = int(df_geocoded['ANO_IS'].max() or 2024)

    # --- Análise Geral ---
    st.header("Análise Geral")
    with st.container():
        selected_states = st.multiselect(
            "Selecionar Estados:", all_states_with_all, default=["Todos"], key="general_states"
        )
        selected_states_filter = get_selected_states(all_states, selected_states)
        year_range = st.slider(
            "Selecionar Período:", min_year, max_year, (min_year, max_year), key="general_year"
        )
        start_year, end_year = year_range

        df_aggregated, df_filtered = aggregate_cases(df_geocoded, selected_states_filter, start_year, end_year)

        if not df_aggregated.empty:
            max_casos = int(df_aggregated['Casos'].max())
            casos_range = st.slider(
                "Filtrar por número de casos:", 1, max_casos, (1, max_casos), key="general_casos"
            )
            min_casos, max_casos = casos_range

            st.subheader("Mapa de Hotspots por Município")
            hotspot_map = create_hotspot_map(df_aggregated, min_casos, max_casos, df_filtered)
            st_folium(hotspot_map or folium.Map(DEFAULT_CENTER, zoom_start=DEFAULT_ZOOM), width=800, height=600)
        else:
            st.warning("Nenhum dado disponível para o mapa de hotspots com os filtros selecionados.")

        st.subheader("Mapa de Densidade de Kernel")
        df_filtered_kde = filter_individual_cases(df_geocoded, selected_states_filter, start_year, end_year)
        with st.expander("Configurações do Mapa de Densidade"):
            blur = st.slider("Nível de Desfoque (Blur):", 5, 50, 20, key="kde_blur")
            radius = st.slider("Raio dos Pontos (Radius):", 10, 50, 25, key="kde_radius")
            gradient_option = st.selectbox(
                "Esquema de Cores:",
                ["Padrão", "Vermelho", "Azul", "Verde"],
                key="kde_gradient"
            )
            gradient_map = {
                "Padrão": None,
                "Vermelho": {0.0: 'blue', 0.4: 'yellow', 1.0: 'red'},
                "Azul": {0.0: 'black', 0.4: 'blue', 1.0: 'white'},
                "Verde": {0.0: 'red', 0.4: 'yellow', 1.0: 'green'}
            }
            selected_gradient = gradient_map[gradient_option]

        kde_map = create_kde_heatmap(df_filtered_kde, blur=blur, radius=radius, gradient=selected_gradient)
        st_folium(kde_map or folium.Map(DEFAULT_CENTER, zoom_start=DEFAULT_ZOOM), width=800, height=600)

    # --- Comparação de Mapas Lado a Lado ---
    st.header("Comparação de Mapas Lado a Lado")
    col1, col2 = st.columns([1, 1], gap="small")

    def render_map(col, map_id, default_year_start, default_year_end, default_states):
        with col:
            st.subheader(f"Mapa {map_id}")
            states = st.multiselect(
                f"Selecionar Estados (Mapa {map_id}):", all_states_with_all,
                default=default_states, key=f"map{map_id}_states"
            )
            states_filter = get_selected_states(all_states, states)
            year_range = st.slider(
                f"Selecionar Período (Mapa {map_id}):", min_year, max_year,
                (default_year_start, default_year_end), key=f"map{map_id}_year"
            )
            start_year, end_year = year_range
            df_agg, df_filt = aggregate_cases(df_geocoded, states_filter, start_year, end_year)

            # Mapa de Hotspots
            st.markdown("**Hotspots por Município**")
            if not df_agg.empty:
                max_casos = int(df_agg['Casos'].max())
                # Define intervalo padrão de 1 a 100, limitado pelo max_casos
                default_casos_max = min(100, max_casos)
                casos_range = st.slider(
                    f"Filtrar por número de casos (Mapa {map_id}):", 1, max_casos,
                    (1, default_casos_max), key=f"map{map_id}_casos"
                )
                hotspot_map = create_hotspot_map(df_agg, *casos_range, df_filt)
                st_folium(
                    hotspot_map or folium.Map(DEFAULT_CENTER, zoom_start=DEFAULT_ZOOM),
                    width=None,
                    height=400
                )
            else:
                st.warning(f"Nenhum dado disponível para o mapa de hotspots do Mapa {map_id}.")

            # Mapa de Densidade de Kernel
            st.markdown("**Densidade de Kernel**")
            df_filtered_kde = filter_individual_cases(df_geocoded, states_filter, start_year, end_year)
            with st.expander(f"Configurações do Mapa de Densidade (Mapa {map_id})"):
                blur = st.slider(
                    "Nível de Desfoque (Blur):", 5, 50, 20, key=f"map{map_id}_kde_blur"
                )
                radius = st.slider(
                    "Raio dos Pontos (Radius):", 10, 50, 25, key=f"map{map_id}_kde_radius"
                )
                gradient_option = st.selectbox(
                    "Esquema de Cores:",
                    ["Padrão", "Vermelho", "Azul", "Verde"],
                    key=f"map{map_id}_kde_gradient"
                )
                gradient_map = {
                    "Padrão": None,
                    "Vermelho": {0.0: 'blue', 0.4: 'yellow', 1.0: 'red'},
                    "Azul": {0.0: 'black', 0.4: 'blue', 1.0: 'white'},
                    "Verde": {0.0: 'red', 0.4: 'yellow', 1.0: 'green'}
                }
                selected_gradient = gradient_map[gradient_option]

            kde_map = create_kde_heatmap(df_filtered_kde, blur=blur, radius=radius, gradient=selected_gradient)
            st_folium(
                kde_map or folium.Map(DEFAULT_CENTER, zoom_start=DEFAULT_ZOOM),
                width=None,
                height=400
            )

    # Mapa 1: Todos os estados, 1994-2016, 1-100 casos
    render_map(col1, 1, 1994, 2016, ["Todos"])
    # Mapa 2: Todos os estados, 2017-2024, 1-100 casos
    render_map(col2, 2, 2017, 2024, ["Todos"])
else:
    st.error("Falha ao carregar os dados geocodificados.")
