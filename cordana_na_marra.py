import pandas as pd

# Caminhos para os arquivos
casos_path = "fa_casoshumanos_1994-2024.csv"
coords_path = "municipio_coordinates.csv"

# Carregar os dados com encoding apropriado
casos = pd.read_csv(casos_path, sep=";", encoding="latin1")
coords = pd.read_csv(coords_path)

# Realizar o merge com base em UF_LPI e MUN_LPI
casos_com_coords = pd.merge(
    casos,
    coords,
    how="left",
    on=["UF_LPI", "MUN_LPI"]
)

# Verificar quantos casos ficaram sem coordenadas
sem_coords = casos_com_coords[casos_com_coords["Latitude"].isna()]
print(f"{len(sem_coords)} casos sem coordenadas encontradas.")

# Salvar o novo dataset
casos_com_coords.to_csv("fa_casoshumanos_1994-2024_com_coords.csv", sep=";", index=False)

print("Novo arquivo salvo como 'fa_casoshumanos_1994-2024_com_coords.csv'")

