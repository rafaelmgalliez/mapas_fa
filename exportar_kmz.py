import pandas as pd
import simplekml
from tqdm import tqdm

# --- Nomes dos arquivos de entrada e saída ---
CASOS_CSV_PATH = "fa_casoshumanos_1994-2025.csv"
COORDENADAS_CSV_PATH = "municipios_coordenadas.csv"
OUTPUT_KMZ_PATH = "animacao_febre_amarela.kmz"

def criar_kmz_animado():
    """
    Lê os dados de casos e coordenadas, combina-os e gera um arquivo KMZ
    com timestamps para animação no Google Earth.
    """
    print("Iniciando o processo de criação do KMZ...")

    # --- Carregar e Preparar os Dados ---
    try:
        print(f"Carregando dados de casos de '{CASOS_CSV_PATH}'...")
        df_casos = pd.read_csv(CASOS_CSV_PATH, sep=',')
        
        print(f"Carregando coordenadas de '{COORDENADAS_CSV_PATH}'...")
        df_coords = pd.read_csv(COORDENADAS_CSV_PATH, sep=';')
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado - {e.filename}")
        print("Por favor, certifique-se de que os arquivos CSV estão na mesma pasta do script.")
        return

    # Renomear colunas para consistência
    df_coords.rename(columns={'latitude': 'Latitude', 'longitude': 'Longitude'}, inplace=True)

    # Combinar os dataframes
    print("Combinando dados de casos e coordenadas...")
    df = pd.merge(df_casos, df_coords, on='MUN_LPI', how='inner')

    # --- Limpeza dos Dados ---
    print("Limpando e formatando os dados...")
    # Converte 'DT_IS' para datetime, tratando erros
    df['DT_IS'] = pd.to_datetime(df['DT_IS'], format='%d/%m/%Y', errors='coerce')
    
    # Remove linhas sem data, latitude ou longitude válidas
    df.dropna(subset=['DT_IS', 'Latitude', 'Longitude'], inplace=True)

    print(f"Total de {len(df)} casos válidos para exportação.")

    # --- Criação do KML ---
    kml = simplekml.Kml(name="Animação de Casos de Febre Amarela")

    # Definir estilos para os pontos (um para óbito, outro para sobrevivente)
    style_obito = simplekml.Style()
    style_obito.iconstyle.color = simplekml.Color.red  # Vermelho para óbitos
    style_obito.iconstyle.scale = 0.7

    style_sobrevivente = simplekml.Style()
    style_sobrevivente.iconstyle.color = simplekml.Color.yellow  # Amarelo para sobreviventes
    style_sobrevivente.iconstyle.scale = 0.6
    
    # --- Iterar e Adicionar Pontos ---
    print("Criando os pontos no arquivo KML...")
    # Usando tqdm para uma barra de progresso
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processando casos"):
        pnt = kml.newpoint()

        # Informações do ponto
        pnt.name = f"Caso em {row['MUN_LPI']}"
        pnt.description = (
            f"<b>Data:</b> {row['DT_IS'].strftime('%d/%m/%Y')}<br>"
            f"<b>Município:</b> {row['MUN_LPI']}, {row['UF_LPI']}<br>"
            f"<b>Idade:</b> {row['IDADE']:.0f} anos<br>"
            f"<b>Sexo:</b> {row['SEXO']}<br>"
            f"<b>Óbito:</b> {'Sim' if row['OBITO'] == 1 else 'Não'}"
        )
        
        # Coordenadas
        pnt.coords = [(row['Longitude'], row['Latitude'])]
        
        # Timestamp (essencial para a animação)
        pnt.timestamp.when = row['DT_IS'].strftime('%Y-%m-%d')
        
        # Aplicar estilo com base no óbito
        if row['OBITO'] == 1:
            pnt.style = style_obito
        else:
            pnt.style = style_sobrevivente

    # --- Salvar o Arquivo ---
    print(f"Salvando o arquivo KMZ em '{OUTPUT_KMZ_PATH}'...")
    # Salva como KMZ (arquivo compactado, melhor para compartilhamento)
    kml.savekmz(OUTPUT_KMZ_PATH)

    print("\nProcesso concluído com sucesso!")
    print(f"Abra o arquivo '{OUTPUT_KMZ_PATH}' no Google Earth Pro para ver a animação.")

if __name__ == "__main__":
    criar_kmz_animado()
