import pandas as pd
from geopy.geocoders import Nominatim
from tqdm import tqdm

# --- Nomes dos arquivos ---
INPUT_CSV_PATH = "fa_epizpnh_1994-2025.csv"
OUTPUT_COORDS_PATH = "epizootias_coordenadas.csv"

def geocodificar_epizootias():
    """
    Lê o arquivo de epizootias, extrai os municípios únicos, busca suas coordenadas
    e salva o resultado em um novo arquivo CSV de coordenadas.
    """
    try:
        # Carrega o CSV de epizootias com o separador correto
        df_epiz = pd.read_csv(INPUT_CSV_PATH, sep=';', encoding='latin-1', on_bad_lines='warn')
        print("Arquivo de epizootias carregado com sucesso.")
    except FileNotFoundError:
        print(f"Erro: O arquivo '{INPUT_CSV_PATH}' não foi encontrado.")
        return
    except Exception as e:
        print(f"Ocorreu um erro ao ler o CSV: {e}")
        return

    # Usa os nomes de coluna corretos
    if 'MUN_OCOR' not in df_epiz.columns or 'UF_OCOR' not in df_epiz.columns:
        print("Erro: As colunas 'MUN_OCOR' e/ou 'UF_OCOR' não foram encontradas no CSV.")
        return
        
    # Combina município e estado para uma busca mais precisa
    df_epiz.dropna(subset=['MUN_OCOR', 'UF_OCOR'], inplace=True)
    df_epiz['MUN_UF'] = df_epiz['MUN_OCOR'] + ', ' + df_epiz['UF_OCOR']
    
    unique_municipios = df_epiz['MUN_UF'].unique()
    print(f"Encontrados {len(unique_municipios)} municípios únicos para geocodificar.")

    geolocator = Nominatim(user_agent="epizootia_geocoder_br_v5")
    coordinates_data = []
    
    for municipio_uf in tqdm(unique_municipios, desc="Geocodificando municípios"):
        try:
            location = geolocator.geocode(municipio_uf, timeout=10)
            # Extrai apenas o nome do município para a chave do dicionário
            municipio_original = municipio_uf.split(',')[0].strip()
            
            if location:
                coordinates_data.append({
                    'MUN_OCOR': municipio_original,
                    'Latitude': location.latitude,
                    'Longitude': location.longitude
                })
            else:
                 coordinates_data.append({
                    'MUN_OCOR': municipio_original,
                    'Latitude': None,
                    'Longitude': None
                })
        except Exception as e:
            print(f"\nErro ao geocodificar '{municipio_uf}': {e}")
            municipio_original = municipio_uf.split(',')[0].strip()
            coordinates_data.append({
                'MUN_OCOR': municipio_original,
                'Latitude': None,
                'Longitude': None
            })
    
    # Cria um DataFrame final apenas com as coordenadas
    df_coords = pd.DataFrame(coordinates_data)
    df_coords.drop_duplicates(subset=['MUN_OCOR'], inplace=True)

    # Salva o arquivo de coordenadas
    df_coords.to_csv(OUTPUT_COORDS_PATH, index=False, sep=';', encoding='utf-8')
    print(f"\nProcesso concluído! Arquivo de coordenadas salvo em '{OUTPUT_COORDS_PATH}'.")

if __name__ == "__main__":
    geocodificar_epizootias()
