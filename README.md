# 📊 Painel Interativo de Febre Amarela: Análise Integrada de Casos Humanos e Epizootias

Este repositório contém os scripts e dados para um painel interativo de análise de Febre Amarela no Brasil, cobrindo o período de 1994 a 2025. O projeto, construído com Streamlit, permite a exploração e comparação lado a lado de dados de **casos humanos** e **epizootias em primatas não humanos (PNH)**, além de análises aprofundadas sobre a letalidade da doença.

## 🚀 Funcionalidades Principais

O painel é organizado em abas para facilitar a navegação:

### Aba 1: Visão Geral Comparativa
* **Mapas Geográficos Sincronizados:** Visualize hotspots e mapas de densidade (kernel) para casos humanos e epizootias. O zoom e o movimento são sincronizados entre os mapas para uma comparação direta.
* **Análise Demográfica:** Um gráfico "Raindrop Plot" (violino + pontos) mostra a distribuição de idade dos casos humanos, segmentado por regiões endêmicas.
* **Séries Temporais:** Gráficos comparativos da ocorrência de casos humanos e epizootias ao longo do tempo, com uma linha de tendência de média móvel de 4 semanas.
* **Análise Sazonal:** Gráficos comparativos do padrão sazonal com uma curva de tendência LOWESS e intervalo de confiança. A suavização da curva é interativamente ajustável por um slider.

### Aba 2: Letalidade por Grupo
* **Comparação Direta:** Compare a taxa de letalidade bruta entre dois grupos (Grupo A vs. Grupo B), definidos por você através da seleção de estados e períodos. O gráfico de barras exibe as taxas com seus respectivos intervalos de confiança de 95%.
* **Análise de Risco Controlado:** Um "Forest Plot" exibe os resultados de um modelo de regressão logística, mostrando o risco relativo (Odds Ratio) de óbito ao comparar os dois grupos, ajustado pelos fatores de confusão de idade e sexo.

### Aba 3: Série Histórica de Letalidade
* **Evolução da Letalidade:** Permite selecionar múltiplos estados e um período para visualizar e comparar a evolução da taxa de letalidade anual em cada um.
* **Tendência com Confiança:** Cada estado selecionado é representado por uma curva de tendência LOWESS e sua respectiva faixa de confiança de 95%, facilitando a identificação de mudanças no perfil de risco ao longo dos anos.

## 📂 Estrutura do Repositório
* **`febre_amarela_mapa.py`**: O script principal que executa o painel interativo com Streamlit.
* **`geocodificar_humanos.py`**: Script de preparação para gerar o arquivo de coordenadas `municipios_coordenadas.csv` a partir dos dados de casos humanos.
* **`geocodificar_epizootias.py`**: Script de preparação para gerar o arquivo de coordenadas `epizootias_coordenadas.csv` a partir dos dados de epizootias.
* **`exportar_kmz.py`**: Script opcional para gerar um arquivo KMZ animado para o Google Earth.
* **`requirements.txt`**: Lista de todas as bibliotecas Python necessárias.
* **`README.md`**: Este arquivo.

## 🛠️ Como Usar

### Pré-requisitos
* Python 3.9+
* Os arquivos de dados `fa_casoshumanos_1994-2025.csv` e `fa_epizpnh_1994-2025.csv` no mesmo diretório.

### Passos para Execução

**1. Clone o Repositório:**
```bash
git clone <URL_DO_SEU_REPOSITORIO>
cd <NOME_DO_SEU_REPOSITORIO>

2. Instale as Dependências:
É altamente recomendável criar um ambiente virtual. Após criar e ativar o ambiente, instale as bibliotecas a partir do arquivo requirements.txt:
Bash

pip install -r requirements.txt

3. Prepare os Dados Geográficos:
Antes da primeira execução do painel, gere os arquivos de coordenadas executando os dois scripts a seguir:
Bash

python geocodificar_humanos.py
python geocodificar_epizootias.py

4. Execute o Painel:
Com tudo pronto, inicie o aplicativo Streamlit:
Bash

streamlit run febre_amarela_mapa.py

Seu navegador abrirá automaticamente com o painel interativo.

📊 Fontes de Dados

Os dados utilizados neste projeto são públicos e foram obtidos do DATASUS, a plataforma de dados abertos do Sistema Único de Saúde (SUS) do Brasil.

    Fonte: OpenDataSUS - Notificações de Febre Amarela

📜 Licença
