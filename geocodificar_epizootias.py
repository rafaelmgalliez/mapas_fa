import pandas as pd
from geopy.geocoders import Nominatim
from tqdm import tqdm
import csv # Usado para detectar o separador

# --- Nomes dos arquivos ---
INPUT_CSV_PATH = "fa_epizpnh_1994-2025.csv"
OUTPUT_COORDS_PATH = "epizootias_coordenadas.csv"

def geocodificar_epizootias():
    """
    Lê o arquivo de epizootias de forma robusta, geocodifica os municípios
    e salva o arquivo de coordenadas separadamente.
    """
    try:
        # --- ABORDAGEM ROBUSTA PARA LEITURA DE CSV ---
        # 1. Tenta detectar o separador
        with open(INPUT_CSV_PATH, 'r', encoding='latin-1') as file:
            try:
                dialect = csv.Sniffer().sniff(file.read(1024))
                sep = dialect.delimiter
                print(f"Separador detectado automaticamente: '{sep}'")
            except csv.Error:
                print("Não foi possível detectar o separador, usando ';' como padrão.")
                sep = ';'
            
        # 2. Lê o CSV usando o motor 'python' que é mais flexível
        df_epiz = pd.read_csv(INPUT_CSV_PATH, sep=sep, encoding='latin-1', on_bad_lines='warn', engine='python')
        print("Arquivo de epizootias carregado com sucesso.")

    except FileNotFoundError:
        print(f"Erro: O arquivo '{INPUT_CSV_PATH}' não foi encontrado.")
        return
    except Exception as e:
        print(f"Ocorreu um erro ao ler o CSV: {e}")
        return

    # --- Verificação de colunas com diagnóstico ---
    required_cols = ['MUN_OCOR', 'UF_OCOR']
    if not all(col in df_epiz.columns for col in required_cols):
        print("\n--- DIAGNÓSTICO DE ERRO ---")
        print("ERRO CRÍTICO: As colunas necessárias não foram encontradas.")
        print("Colunas que foram encontradas no arquivo:")
        print(df_epiz.columns)
        print("---------------------------\n")
        print("Isso geralmente significa que o separador (sep) está incorreto, mesmo após a detecção.")
        print("Por favor, verifique o arquivo CSV em um editor de texto simples.")
        return
        
    # Combina município e estado para uma busca mais precisa
    df_epiz.dropna(subset=['MUN_OCOR', 'UF_OCOR'], inplace=True)
    df_epiz['MUN_UF'] = df_epiz['MUN_OCOR'] + ', ' + df_epiz['UF_OCOR']
    
    unique_municipios = df_epiz['MUN_UF'].unique()
    print(f"Encontrados {len(unique_municipios)} municípios únicos para geocodificar.")

    geolocator = Nominatim(user_agent="epizootia_geocoder_br_v6")
    coordinates_data = []
    
    for municipio_uf in tqdm(unique_municipios, desc="Geocodificando municípios"):
        try:
            location = geolocator.geocode(municipio_uf, timeout=10)
            municipio_original = municipio_uf.split(',')[0].strip()
            
            if location:
                coordinates_data.append({'MUN_OCOR': municipio_original, 'Latitude': location.latitude, 'Longitude': location.longitude})
            else:
                coordinates_data.append({'MUN_OCOR': municipio_original, 'Latitude': None, 'Longitude': None})
        except Exception as e:
            print(f"\nErro ao geocodificar '{municipio_uf}': {e}")
            coordinates_data.append({'MUN_OCOR': municipio_original.split(',')[0].strip(), 'Latitude': None, 'Longitude': None})
    
    df_coords = pd.DataFrame(coordinates_data)
    df_coords.drop_duplicates(subset=['MUN_OCOR'], inplace=True)

    df_coords.to_csv(OUTPUT_COORDS_PATH, index=False, sep=';', encoding='utf-8')
    print(f"\nProcesso concluído! Arquivo de coordenadas salvo em '{OUTPUT_COORDS_PATH}'.")

if __name__ == "__main__":
    geocodificar_epizootias()
