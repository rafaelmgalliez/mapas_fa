# Mapas Interativos de Casos de Febre Amarela

Este repositório contém o código Python para gerar mapas interativos de casos de febre amarela, utilizando as bibliotecas Streamlit e Folium. O objetivo é visualizar a distribuição geográfica e a ocorrência temporal dos casos, permitindo uma análise exploratória dos dados.

## Visão Geral

O projeto oferece duas formas principais de visualização:

1.  **Mapa de Hotspots por Município:** Exibe círculos nos municípios onde ocorreram casos de febre amarela. O tamanho e a cor dos círculos são proporcionais à quantidade de casos registrados em cada município dentro do período e estados selecionados.

2.  **Mapa de Densidade de Kernel de Casos Individuais:** Mostra a densidade dos casos individuais de febre amarela como um mapa de calor, permitindo identificar áreas com maior concentração de ocorrências.

## Funcionalidades

* **Seleção de Estados:** Permite ao usuário selecionar um ou mais estados brasileiros para filtrar os dados exibidos nos mapas. Inclui a opção de selecionar "Todos" os estados.
* **Seleção de Período:** Permite ao usuário definir um intervalo de anos para visualizar os casos.
* **Filtragem por Quantidade de Casos por Município (Mapa de Hotspots):** Permite ao usuário filtrar os municípios exibidos no mapa de hotspots com base no número de casos ocorridos (de 1 ao máximo de casos em um município).
* **Visualização Interativa:** Os mapas gerados com Folium são interativos, permitindo zoom, pan e tooltips com informações sobre os casos ou municípios.
* **Carregamento de Dados:** O código carrega os dados de casos de febre amarela de um arquivo CSV (com coordenadas pré-calculadas).

## Como Usar

1.  **Pré-requisitos:**
    * Python 3.6 ou superior
    * As seguintes bibliotecas Python instaladas:
        ```bash
        pip install streamlit pandas folium streamlit-folium matplotlib biopython geopy
        ```
    * Um arquivo CSV contendo os dados de casos de febre amarela com as seguintes colunas (mínimo):
        * `UF_LPI`: Unidade Federativa (sigla do estado).
        * `MUN_LPI`: Município de ocorrência.
        * `DT_IS`: Data de início dos sintomas (formato `dd/mm/YYYY`).
        * `ANO_IS`: Ano de início dos sintomas (geralmente extraído de `DT_IS`).
        * `Latitude`: Latitude da ocorrência (ou do município, se agregado).
        * `Longitude`: Longitude da ocorrência (ou do município, se agregado).
        * `Casos` (opcional, para o mapa de hotspots agregado).

2.  **Execução:**
    * Clone este repositório (se disponível).
    * Certifique-se de que o arquivo de dados CSV (`fa_casoshumanos_1994-2024_com_coords.csv` por padrão) esteja no mesmo diretório do script Python (`febre_amarela_mapa.py`).
    * Abra o terminal ou prompt de comando, navegue até o diretório do projeto e execute o seguinte comando:
        ```bash
        streamlit run febre_amarela_mapa.py
        ```
    * O aplicativo será aberto automaticamente no seu navegador web.

3.  **Interação:**
    * Utilize os controles na barra lateral para selecionar os estados e o período desejado.
    * Para o mapa de hotspots, use o slider para filtrar os municípios com base na quantidade de casos.
    * Interaja com os mapas utilizando o mouse para zoom e pan. Passe o mouse sobre os marcadores para ver informações adicionais.

## Arquivos no Repositório

* `febre_amarela_mapa.py`: O script Python principal que gera o aplicativo Streamlit.
* `fa_casoshumanos_1994-2024_com_coords.csv` (exemplo): Um arquivo CSV de dados de casos de febre amarela com colunas de latitude e longitude (pode ter um nome diferente no seu caso).
* `README.md`: Este arquivo com a descrição do projeto.
* `generate_coord.py` (opcional): Um script para geocodificar dados brutos de casos (se o arquivo CSV com coordenadas não estiver pronto).
* Outros arquivos auxiliares (se houver).

## Notas

* A precisão dos mapas depende da qualidade e precisão dos dados de latitude e longitude no arquivo CSV.
* Para grandes volumes de dados, o carregamento inicial e a renderização dos mapas podem levar algum tempo.
* O código pode ser adaptado para carregar dados de diferentes fontes ou para adicionar mais funcionalidades de visualização e análise.

## Contribuições

Contribuições para este projeto são bem-vindas. Sinta-se à vontade para abrir issues para relatar bugs ou sugerir melhorias, ou enviar pull requests com suas modificações.

## Licença

[Aqui você pode adicionar a licença sob a qual o projeto está distribuído, por exemplo, MIT License]
