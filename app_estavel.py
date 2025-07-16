import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, DualMap
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint
import statsmodels.formula.api as smf

# --- Nomes dos arquivos de dados ---
CASOS_HUMANOS_PATH = "fa_casoshumanos_1994-2025.csv"
COORDENADAS_HUMANOS_PATH = "municipios_coordenadas.csv"
EPIZOOTIAS_PATH = "fa_epizpnh_1994-2025.csv"
COORDENADAS_EPIZOOTIAS_PATH = "epizootias_coordenadas.csv"

# --- Funções de Carregamento ---
@st.cache_data
def load_and_merge_data(data_path, coords_path, is_epizootia=False):
    try:
        sep_data = ';' if is_epizootia else ','
        df_data = pd.read_csv(data_path, sep=sep_data, encoding='latin-1', on_bad_lines='warn', engine='python')
        df_coords = pd.read_csv(coords_path, sep=';', encoding='latin-1')
    except FileNotFoundError as e:
        st.error(f"Arquivo não encontrado: {e.filename}. Por favor, execute os scripts de geocodificação primeiro.")
        return None

    join_key_data = 'MUN_OCOR' if is_epizootia else 'MUN_LPI'
    join_key_coords = 'MUN_OCOR' if is_epizootia else 'MUN_LPI'
    df_merged = pd.merge(df_data, df_coords, left_on=join_key_data, right_on=join_key_coords, how='inner')

    if is_epizootia:
        df_merged.rename(columns={'MUN_OCOR': 'MUN_LPI', 'UF_OCOR': 'UF_LPI', 'DATA_OCOR': 'DT_IS', 'ANO_OCOR': 'ANO_IS'}, inplace=True)
        df_merged['Endemico'] = -1; df_merged['IDADE'] = -1
    else:
        if 'latitude' in df_merged.columns: df_merged.rename(columns={'latitude': 'Latitude', 'longitude': 'Longitude'}, inplace=True)

    df_merged['DT_IS'] = pd.to_datetime(df_merged['DT_IS'], errors='coerce', dayfirst=True)
    df_merged.dropna(subset=['DT_IS', 'Latitude', 'Longitude'], inplace=True)
    return df_merged

# --- Funções de Plotagem ---
def create_synced_hotspot_maps(df_humanos_agg, df_epizootias_agg):
    center_lat, center_lon = -14, -51
    if not df_humanos_agg.empty: center_lat, center_lon = df_humanos_agg['Latitude'].mean(), df_humanos_agg['Longitude'].mean()
    m = DualMap(location=[center_lat, center_lon], zoom_start=4)
    cmap_h = LinearSegmentedColormap.from_list("humanos_cmap", ["#FFFF00", "#FF0000"])
    if not df_humanos_agg.empty:
        max_casos_h = df_humanos_agg['Casos'].max()
        for _, row in df_humanos_agg.iterrows():
            radius = np.sqrt(row['Casos']) * 3; color_val = row['Casos'] / max_casos_h if max_casos_h > 0 else 0; marker_color = matplotlib.colors.to_hex(cmap_h(color_val))
            folium.CircleMarker(location=[row['Latitude'], row['Longitude']], radius=radius, popup=f"<b>{row['MUN_LPI']}</b><br>{int(row['Casos'])} casos humanos", color=marker_color, fill=True, fill_color=marker_color, fill_opacity=0.7).add_to(m.m1)
    cmap_e = LinearSegmentedColormap.from_list("epiz_cmap", ["#00FFFF", "#0000FF"])
    if not df_epizootias_agg.empty:
        max_casos_e = df_epizootias_agg['Casos'].max()
        for _, row in df_epizootias_agg.iterrows():
            radius = np.sqrt(row['Casos']) * 3; color_val = row['Casos'] / max_casos_e if max_casos_e > 0 else 0; marker_color = matplotlib.colors.to_hex(cmap_e(color_val))
            folium.CircleMarker(location=[row['Latitude'], row['Longitude']], radius=radius, popup=f"<b>{row['MUN_LPI']}</b><br>{int(row['Casos'])} epizootias", color=marker_color, fill=True, fill_color=marker_color, fill_opacity=0.7).add_to(m.m2)
    return m

def create_synced_kde_maps(df_humanos, df_epizootias):
    center_lat, center_lon = -14, -51
    if not df_humanos.empty: center_lat, center_lon = df_humanos['Latitude'].mean(), df_humanos['Longitude'].mean()
    m = DualMap(location=[center_lat, center_lon], zoom_start=4)
    if not df_humanos.empty: HeatMap(df_humanos[['Latitude', 'Longitude']].values, radius=15, blur=20).add_to(m.m1)
    if not df_epizootias.empty: HeatMap(df_epizootias[['Latitude', 'Longitude']].values, radius=15, blur=20, gradient={0.4: 'cyan', 0.65: 'blue', 1: 'darkblue'}).add_to(m.m2)
    return m

def create_raincloud_plot(df):
    if df.empty or 'IDADE' not in df.columns or df['IDADE'].iloc[0] == -1: return None
    df_plot = df.dropna(subset=['IDADE', 'Endemico']).copy(); df_plot['Região Endêmica'] = df_plot['Endemico'].map({1.0: 'Sim', 0.0: 'Não'}); df_plot['dummy'] = ""
    fig, ax = plt.subplots(figsize=(12, 5)); sns.set_style("whitegrid"); pal = {"Sim": "#3498db", "Não": "#e74c3c"}
    ax = sns.violinplot(data=df_plot, x='IDADE', y='dummy', hue='Região Endêmica', split=True, inner='box', palette=pal, ax=ax, linewidth=1.5, cut=0)
    sns.stripplot(data=df_plot, x='IDADE', y='dummy', hue='Região Endêmica', jitter=0.15, ax=ax, dodge=True, palette=pal, size=3, alpha=0.5)
    handles, labels = ax.get_legend_handles_labels(); ax.legend(handles[:2], labels[:2], title='Região Endêmica')
    ax.set_title('Distribuição de Idade por Região Endêmica', fontsize=16, pad=20, weight='bold'); ax.set_xlabel('Idade (anos)'); ax.set_ylabel(''); ax.set_yticks([]); sns.despine(left=True); plt.tight_layout(); return fig

def create_time_series_plot(df, title):
    if df.empty: fig, ax = plt.subplots(figsize=(10, 4)); ax.text(0.5, 0.5, 'Sem dados para exibir', ha='center', va='center'); return fig
    weekly_cases = df.set_index('DT_IS').resample('W-Mon').size(); moving_avg = weekly_cases.rolling(window=4, center=True).mean()
    fig, ax = plt.subplots(figsize=(10, 4)); ax.bar(weekly_cases.index, weekly_cases.values, color='lightblue', alpha=0.7, label='Casos Semanais', width=7); ax.plot(moving_avg.index, moving_avg.values, color='darkblue', linewidth=2, label='Média Móvel (4 sem.)')
    ax.set_xlim([df['DT_IS'].min(), df['DT_IS'].max()]); ax.set_title(title, fontsize=14, weight='bold'); ax.set_ylabel('Nº de Ocorrências'); ax.legend(); sns.despine(); plt.tight_layout(); return fig

def create_seasonal_plot_from_scratch(df, title, frac_value):
    if len(df) < 3: fig, ax = plt.subplots(figsize=(10, 4)); ax.text(0.5, 0.5, 'Sem dados para exibir', ha='center', va='center'); return fig
    df_seasonal = df.copy(); df_seasonal['Mês'] = df_seasonal['DT_IS'].dt.month; monthly_cases = df_seasonal.groupby('Mês').size().reset_index(name='Casos')
    if len(monthly_cases) < 3: fig, ax = plt.subplots(figsize=(10, 4)); ax.text(0.5, 0.5, 'Sem dados para exibir', ha='center', va='center'); return fig
    x = monthly_cases['Mês'].values; y = monthly_cases['Casos'].values
    plt.style.use('seaborn-v0_8-whitegrid'); fig, ax = plt.subplots(figsize=(12, 6)); n_bootstraps = 200; bootstrapped_lowess = []; lowess_x_grid = np.linspace(min(x), max(x), 100)
    for _ in range(n_bootstraps):
        sample_indices = np.random.choice(len(x), len(x), replace=True); x_sample, y_sample = x[sample_indices], y[sample_indices]; sort_order = np.argsort(x_sample); x_sample_sorted, y_sample_sorted = x_sample[sort_order], y_sample[sort_order]
        lowess_sample = sm.nonparametric.lowess(y_sample_sorted, x_sample_sorted, frac=frac_value); interp_func = np.interp(lowess_x_grid, lowess_sample[:, 0], lowess_sample[:, 1]); bootstrapped_lowess.append(interp_func)
    lower_bound = np.percentile(bootstrapped_lowess, 2.5, axis=0); upper_bound = np.percentile(bootstrapped_lowess, 97.5, axis=0)
    ax.fill_between(lowess_x_grid, lower_bound, upper_bound, color='dodgerblue', alpha=0.2, zorder=1)
    main_lowess = sm.nonparametric.lowess(y, x, frac=frac_value); y_pred_interp = np.interp(lowess_x_grid, main_lowess[:, 0], main_lowess[:, 1])
    ax.plot(lowess_x_grid, y_pred_interp, color='blue', linewidth=2.5, zorder=3); ax.scatter(x, y, color='#333333', alpha=0.6, s=50, zorder=2)
    ax.set_title(title, fontsize=14, weight='bold'); ax.set_xlabel("Mês", fontsize=12); ax.set_ylabel("Número de Casos", fontsize=12)
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'], rotation=45, ha='right')
    ax.set_ylim(0); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); plt.tight_layout(); return fig

def create_lethality_barchart(df_A, df_B, title_A, title_B):
    rates = {}
    for name, df_group in [(title_A, df_A), (title_B, df_B)]:
        if not df_group.empty and 'OBITO' in df_group.columns:
            n_cases = len(df_group); n_deaths = int(df_group['OBITO'].sum())
            if n_cases > 0:
                rate = n_deaths / n_cases; ci_low, ci_upp = proportion_confint(count=n_deaths, nobs=n_cases, method='wilson')
                rates[name] = {'rate': rate, 'ci_low': ci_low, 'ci_upp': ci_upp, 'label': f"{name}\n(N={n_cases})"}
    if not rates: fig, ax = plt.subplots(); ax.text(0.5, 0.5, 'Dados insuficientes.', ha='center'); return fig
    fig, ax = plt.subplots(figsize=(8, 5)); labels = list(rates.keys()); lethality_rates = [d['rate'] for d in rates.values()]; errors = [[d['rate'] - d['ci_low'] for d in rates.values()], [d['ci_upp'] - d['rate'] for d in rates.values()]]
    bars = ax.bar(labels, lethality_rates, yerr=errors, capsize=5, color=['skyblue', 'salmon'])
    ax.set_ylabel("Taxa de Letalidade (%)"); ax.set_title("Comparação da Taxa de Letalidade Bruta", weight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format)); ax.set_ylim(0, max(lethality_rates) * 1.5 if lethality_rates else 1)
    for bar in bars:
        height = bar.get_height(); ax.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    sns.despine(); plt.tight_layout(); return fig

def create_lethality_forest_plot(df, title):
    df_model = df.dropna(subset=['OBITO', 'IDADE', 'SEXO', 'Grupo']).copy(); df_model = df_model[df_model['SEXO'].isin(['M', 'F'])]; df_model['OBITO'] = df_model['OBITO'].astype(int)
    if len(df_model) < 20 or len(df_model['OBITO'].unique()) < 2 or len(df_model['Grupo'].unique()) < 2:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, 'Dados insuficientes para o modelo.', ha='center'); return fig
    try:
        model = smf.logit('OBITO ~ IDADE + C(SEXO) + C(Grupo, Treatment(reference="Grupo A"))', data=df_model).fit(disp=0)
        params = pd.DataFrame(np.exp(model.params), columns=['Odds Ratio']); conf = np.exp(model.conf_int()); conf.columns = ['CI Lower', 'CI Upper']
        odds_ratios = params.join(conf); odds_ratios = odds_ratios.drop('Intercept')
    except Exception:
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, 'Não foi possível ajustar o modelo.', ha='center'); return fig
    fig, ax = plt.subplots(figsize=(8, 5)); y_pos = np.arange(len(odds_ratios))
    ax.errorbar(x=odds_ratios['Odds Ratio'], y=y_pos, xerr=[odds_ratios['Odds Ratio'] - odds_ratios['CI Lower'], odds_ratios['CI Upper'] - odds_ratios['Odds Ratio']], fmt='o', color='black', capsize=5)
    ax.axvline(x=1, linestyle='--', color='grey'); ax.set_yticks(y_pos); ax.set_yticklabels(odds_ratios.index); ax.set_xlabel("Odds Ratio (Risco de Óbito)"); ax.set_title(title, weight='bold'); sns.despine(); plt.tight_layout(); return fig

# --- Interface do Streamlit ---
st.set_page_config(layout="wide"); st.title("Análise de Febre Amarela: Casos Humanos e Epizootias")
df_humanos = load_and_merge_data(CASOS_HUMANOS_PATH, COORDENADAS_HUMANOS_PATH, is_epizootia=False)
df_epizootias = load_and_merge_data(EPIZOOTIAS_PATH, COORDENADAS_EPIZOOTIAS_PATH, is_epizootia=True)

if df_humanos is not None and df_epizootias is not None:
    tab1, tab2 = st.tabs(["Visão Geral Comparativa", "Análise de Letalidade (Humanos)"])
    
    with tab1:
        st.sidebar.header("⚙️ Filtros para Visão Geral")
        all_ufs_geral = sorted(pd.concat([df_humanos['UF_LPI'], df_epizootias['UF_LPI']]).unique())
        min_year_geral = int(min(df_humanos['ANO_IS'].min(), df_epizootias['ANO_IS'].min()))
        max_year_geral = int(max(df_humanos['ANO_IS'].max(), df_epizootias['ANO_IS'].max()))
        ufs_selecionadas = st.sidebar.multiselect("Selecione UF", all_ufs_geral, default=all_ufs_geral, key="ufs_geral")
        year_range = st.sidebar.slider("Selecione Período", min_year_geral, max_year_geral, (min_year_geral, max_year_geral), key="anos_geral")
        df_humanos_filt = df_humanos[(df_humanos['ANO_IS'].between(year_range[0], year_range[1])) & (df_humanos['UF_LPI'].isin(ufs_selecionadas))]
        df_epizootias_filt = df_epizootias[(df_epizootias['ANO_IS'].between(year_range[0], year_range[1])) & (df_epizootias['UF_LPI'].isin(ufs_selecionadas))]
        st.sidebar.markdown("---"); st.sidebar.subheader("Filtro Adicional (Casos Humanos)")
        df_agg_humanos = df_humanos_filt.groupby(['MUN_LPI', 'Latitude', 'Longitude']).size().reset_index(name='Casos')
        df_humanos_final = df_humanos_filt
        if not df_agg_humanos.empty:
            max_casos_h = int(df_agg_humanos['Casos'].max()) if df_agg_humanos['Casos'].max() > 1 else 2
            min_casos_filtro = st.sidebar.slider("Nº mínimo de casos humanos", 1, max_casos_h, 1)
            df_agg_humanos = df_agg_humanos[df_agg_humanos['Casos'] >= min_casos_filtro]
            munis_humanos_filtrados = df_agg_humanos['MUN_LPI'].unique()
            df_humanos_final = df_humanos_filt[df_humanos_filt['MUN_LPI'].isin(munis_humanos_filtrados)]
        st.sidebar.markdown("---"); st.sidebar.subheader("Ajuste do Gráfico Sazonal")
        lowess_frac = st.sidebar.slider("Nível de Suavização (Sazonal)", min_value=0.10, max_value=1.0, value=0.4, step=0.05)
        df_agg_epiz = df_epizootias_filt.groupby(['MUN_LPI', 'Latitude', 'Longitude']).size().reset_index(name='Casos')
        st.header("🗺️ Análise Geográfica Sincronizada"); st_folium(create_synced_hotspot_maps(df_agg_humanos, df_agg_epiz), key="synced_hotspots", width=1400, height=500)
        st.markdown("<br>", unsafe_allow_html=True); st_folium(create_synced_kde_maps(df_humanos_final, df_epizootias_filt), key="synced_kde", width=1400, height=500)
        st.markdown("---"); st.header("🧑 Análise Demográfica (Apenas Casos Humanos)"); st.pyplot(create_raincloud_plot(df_humanos_final))
        st.markdown("---"); st.header("⏳ Análise Temporal e Sazonal Comparativa")
        col_ts1, col_ts2 = st.columns(2)
        with col_ts1: st.pyplot(create_time_series_plot(df_humanos_filt, "Série Temporal: Casos Humanos"))
        with col_ts2: st.pyplot(create_time_series_plot(df_epizootias_filt, "Série Temporal: Epizootias"))
        col_saz1, col_saz2 = st.columns(2)
        with col_saz1: st.pyplot(create_seasonal_plot_from_scratch(df_humanos_filt, "Sazonalidade: Casos Humanos", lowess_frac))
        with col_saz2: st.pyplot(create_seasonal_plot_from_scratch(df_epizootias_filt, "Sazonalidade: Epizootias", lowess_frac))

    with tab2:
        st.header("🔬 Análise Comparativa de Letalidade")
        st.info("Defina dois grupos (por UF e período) para comparar a taxa de letalidade e o risco relativo.")
        
        all_ufs_lethality = sorted(df_humanos['UF_LPI'].unique())
        min_year_lethality = int(df_humanos['ANO_IS'].min()); max_year_lethality = int(df_humanos['ANO_IS'].max())
        
        colA, colB = st.columns(2)
        with colA:
            st.markdown("##### Definição do Grupo A")
            ufs_A = st.multiselect("Estados (Grupo A)", all_ufs_lethality, default=['SP', 'MG'], key="ufs_A_lethality")
            years_A = st.slider("Período (Grupo A)", min_year_lethality, max_year_lethality, (2016, 2018), key="years_A_lethality")
        with colB:
            st.markdown("##### Definição do Grupo B")
            ufs_B = st.multiselect("Estados (Grupo B)", all_ufs_lethality, default=all_ufs_lethality, key="ufs_B_lethality")
            years_B = st.slider("Período (Grupo B)", min_year_lethality, max_year_lethality, (2000, 2010), key="years_B_lethality")
            
        df_grupo_A = df_humanos[(df_humanos['ANO_IS'].between(years_A[0], years_A[1])) & (df_humanos['UF_LPI'].isin(ufs_A))].copy()
        df_grupo_B = df_humanos[(df_humanos['ANO_IS'].between(years_B[0], years_B[1])) & (df_humanos['UF_LPI'].isin(ufs_B))].copy()
        
        col_leth1, col_leth2 = st.columns(2)
        with col_leth1:
            st.markdown("###### Taxa de Letalidade Bruta com IC 95%")
            st.pyplot(create_lethality_barchart(df_grupo_A, df_grupo_B, "Grupo A", "Grupo B"))
        with col_leth2:
            st.markdown("###### Risco Relativo (Controlado por Idade e Sexo)")
            df_grupo_A['Grupo'] = 'Grupo A'; df_grupo_B['Grupo'] = 'Grupo B'
            df_comparison = pd.concat([df_grupo_A, df_grupo_B])
            st.pyplot(create_lethality_forest_plot(df_comparison, "Risco de Óbito (Odds Ratio)"))
else:
    st.error("Falha ao carregar os dados. Verifique se os arquivos CSV estão na pasta correta e se os scripts de geocodificação foram executados.")
