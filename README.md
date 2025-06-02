# Método Objetivo para Avaliação Quantitativa de Algoritmos de Pitch Shifting

Uma ferramenta de linha de comando para avaliar quantitativamente algoritmos de pitch shifting sobre arquivos de áudio.
Calcula métricas analíticas (ex. taxa de cruzamentos por zero, cenróide espectral, caracteristicas tonais) e métricas comparativas (ex. distância MFCC, desvio de afinação, correlação cruzada) entre arquivos WAV de entrada e vários arquivos de saída gerados por diferentes plugins ou algoritmos. Gera relatórios JSON/CSV e gráficos para ajudar a determinar qual implementação de pitch shifter melhor preserva a qualidade do áudio.

---

## Sumário

1. [Funcionalidades](#funcionalidades)
2. [Pré requisitos](#pré-requisitos)
3. [Estrutura do projeto](#estrutura-do-projeto)
4. [Instalação](#instalação)
5. [Uso](#uso)

   * [Argumentos de Linha de Comando](#argumentos-de-linha-de-comando)
   * [Exemplo](#exemplo)
6. [Docker](#docker)
7. [Dados de entrada](#dados-de-entrada)
8. [Dados de saída](#dados-de-saída)

   * [Espectrogramas](#espectrogramas)
   * [Métricas JSON/CSV](#métricas-jsoncsv)
   * [Gráficos](#gráficos)
   * [Ranque de plugins](#ranque-de-plugins)
9. [Descrição de funções](#descrição-de-funções)

   * [`analytical_metrics()`](#analytical_metrics)
   * [`comparative_metrics()`](#comparative_metrics)
   * [`plot_spectrograms()`](#plot_spectrograms)
   * [`calculate_metrics()`](#calculate_metrics)
   * [`evaluate_errors()`](#evaluate_errors)
10. [Autores & Contato](#autores--contato)

---

## Funcionalidades

* **Visualização de Espectrogramas**
  Plota epectrogramas para cada arquivo WAV original lado a lado com os correspondentes a todas as saídas transpostos (para cada plugins/razão) em uma mesma figura.

* **Métricas Analíticas**

  * Taxa de Cruzamentos por Zero (ZCR)
  * Energia Absoluta do Áudio (soma de quadrados)
  * Medida de Planicidade Espectral (SFM)
  * Tonalidade (via componente harmônico de HPSS)
  * Frequência Fundamental Média (média YIN f0)
  * Centróide Espectral
  * Desvio Médio Absoluto de Frequência (MAFD)
  * Número de Picos Senoidais (via detecção de picos em espectrograma)

* **Métricas Comparativas**

  * Distância L1 de MFCC
  * Distância de Sonoridade (diferença RMS quadro a quadro)
  * Desvio de Afinação (erro médio absoluto de pitch via PYIN)
  * Correlação Cruzada Máxima (similaridade no domínio do tempo)
  * Compensação de Desvio de Fase
  * Distância de Reinicialização de Fase (descontinuidades instantânea em fase)

* **Análise de Erro & Normalização**
  Calcula o erro relativo para cada métrica:

  * Métricas relacionadas a frequência escalam com a razão de pitch.
  * Métricas relacionadas a energia se mantém constantes.
  * Métricas comparativas comparam sinais transpostos e os originais correspondentes.

* **Relatórios conjuntos & Ranqueamento por Z-score composto**

  * Arquivos JSON por input e por plugin contendo resultados reais.
  * Arquivos CSV para métricas sobre sinais originais e médias por plugin.
  * Normalização Z-score sobre todas as métricas para ranquear a performance dos plugins.
  * Múltiplos gráficos de resumo.

* **Interface em Linha de Comando (CLI)**
  Ponto de entrada único `mopaqdaps` que automatiza o fluxo inteiro.

---

## Pré requisitos

* Python ≥ 3.7
* librosa ≥ 0.9.2
* numpy ≥ 1.19.5
* pandas ≥ 1.1.5
* matplotlib ≥ 3.2.2
* seaborn ≥ 0.11.0
* scipy ≥ 1.5.4
* scikit-learn ≥ 0.24.0
* torch ≥ 1.9.0
* torchaudio ≥ 0.9.0

Essas dependências são listadas e instaladas autmaticamente via `pip install .` (vide Instalação).

---

## Estrutura do projeto

```
MOPAQDAPS/
├── mopaqdaps/                    # Package source code
│   ├── __init__.py
│   └── main.py                   # All core logic and CLI entry point
│
├── README.md                     # This documentation
├── requirements.txt              # External dependencies
├── setup.py                      # setuptools configuration
└── MANIFEST.in                   # Include README in the package
```

Após instalação, executar `mopaqdaps` (instalado nas variáveis PATH).

---

## Instalação

1. **Clonar ou baixar este repositório**:

   ```bash
   git clone https://github.com/jpTorres-13/MOPAQDAPS.git
   cd MOPAQDAPS
   ```

2. **Criar ambiente virtual** (opcional recomendado):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instalar pacote** (do diretório contendo `setup.py`):

   ```bash
   pip install .
   ```

   Isso instala:

   * O script `mopaqdaps` para CLI
   * Todas as dependências listadas em `requirements.txt`

> **Nota**: Se você já tem um ambiente python adequado com as dependências, pode pular a criação de ambiente virtual.

---

## Uso

Uma vez instalado, o ponto de entrada CLI `mopaqdaps` está disponível nas variáveis PATH.

```bash
mopaqdaps --dataset_root <DATA_IN> [--results_root <DATA_OUT>] [--skip_spectrograms]
```

### Argumentos de linha de comando

* `--dataset_root`, `-d` (mandatório)
  Caminho para o diretório do **dataset**, que precisa conter as pastas:

  ```
  <DATA_IN>/
  ├── input/    # Arquivos WAV originais (e.g. “C.wav”, “D.wav”, etc.)
  └── output/   # Sub-pastas específicas para cada plugin, cada uma contendo as pastas específicas para as razões de transposição
                # ex. output/pluginA/ratio-2.0/*.wav, output/pluginB/ratio-4.0/*.wav, etc.
  ```

* `--results_root`, `-r` (default: `results`)
  Diretório onde todas as saídas serão escritas. A estrutura seguinte será criada sob essa pasta:

  ```
  <DATA_OUT>/
  ├── spectrograms/             # Arquivos PNG com os espectrogramas para cada input+plugin
  ├── metrics/
  │   ├── analytical/           # rquivos JSON por input para métricas analíticas
  │   └── comparison/           # rquivos JSON por input para métricas comparativas
  ├── results/                  # Arquivos JSON conjuntos e CSVs
  └── plots/                    # Gráficos (z-score charts, boxplots, etc.)
  ```

* `--skip_spectrograms`
  Se especificado, pula a geração dos espectrogramas. Útil se você já os gerou em execuções anteriores ou precisa poupar tempo.

### Exemplo

Assumindo que seu conjunto de dados esteja estruturado da seguinte forma:

```
./MOPAQDAPS/data_in/
├── input/
│   ├── C.wav
│   ├── D.wav
│   └── ... (other original .wav files)
└── output/
    ├── pluginA/
    │   ├── ratio-2.0/
    │   │   ├── C.wav
    │   │   └── D.wav
    │   └── ratio-4.0/
    │       ├── C.wav
    │       └── D.wav
    ├── pluginB/
    │   ├── ratio-2.0/
    │   │   ├── C.wav
    │   │   └── D.wav
    │   └── ratio-4.0/
    │       ├── C.wav
    │       └── D.wav
    └── ... (other plugins)
```

Execute:

```bash
mopaqdaps \
  --dataset_root ./MOPAQDAPS/data_in \
  --results_root ./MOPAQDAPS/data_out
```

Isso irá gerar:

```
./MOPAQDAPS/data_out
├── spectrograms/
│   ├── C_pluginA_all_ratios.png
│   ├── C_pluginB_all_ratios.png
│   ├── D_pluginA_all_ratios.png
│   └── D_pluginB_all_ratios.png
│
├── metrics/
│   ├── analytical/
│   │   ├── C/
│   │   │   ├── ratio-2.0.json
│   │   │   └── ratio-4.0.json
│   │   └── D/
│   │       ├── ratio-2.0.json
│   │       └── ratio-4.0.json
│   └── comparison/
│       ├── C/
│       │   ├── ratio-2.0.json
│       │   └── ratio-4.0.json
│       └── D/
│           ├── ratio-2.0.json
│           └── ratio-4.0.json
│
├── results/
│   ├── original.json
│   ├── pluginA.json
│   ├── pluginB.json
│   ├── original_metrics.csv
│   ├── original_metrics_zscore.json
│   ├── metrics_mean_comparison.csv
│   ├── plugin_composite_ranking.csv
│   └── ... (other summary JSON/CSV files)
│
└── plots/
    ├── zscore_by_input_*.png
    ├── zscore_by_ratio_*.png
    ├── zscore_distribution_per_plugin.png
    ├── best_plugin_per_input_count.png
    └── composite_score_ranking.png
```

Se você desejar pular a geração de espectrogramas, inclua:

```bash
mopaqdaps \
  --dataset_root ./MOPAQDAPS/data_in \
  --results_root ./MOPAQDAPS/data_out
  --skip_spectrograms
```

---

## Docker

Para executar via Docker, siga os passos abaixo:

1. **Assegure-se** de que você tem Docker instalado em seu sistema.

2. **Mantenha** o arquivo `Dockerfile` na raiz do projeto (junto a `setup.py` e `requirements.txt`):

3. Execute o seguinte comando para **gerar a imagem**:

   ```bash
   docker build -t mopaqdaps .
   ```

4. **Execute** o container, montando o dataset e o diretório de saída. Exemplo:

   ```bash
   docker run --rm \
     -v /path/to/your/data_in:/data:ro \
     -v /path/to/local/data_out:/out \
     mopaqdaps \
     --dataset_root /data_in \
     --results_root /data_out
   ```

   * `-v /path/to/your/data_in:/data_in:ro` monta o seu diretório local de dataset (contendo `input/` e `output/`) como `/data_in` dentro do container (read-only).
   * `-v /path/to/local/data_out:/data_out` monta o seu diretório local de saída como `/data_out` dentros do container (read/write).
   * O comando `mopaqdaps --dataset_root /data_in --results_root /data_out` executa o MOPAQDAPS dentro do container.

5. **(Opcional)** Se desejar pular a geração de espectrogramas:

   ```bash
   docker run --rm \
     -v /path/to/your/data_in:/data:ro \
     -v /path/to/local/data_out:/out \
     mopaqdaps \
     --dataset_root /data_in \
     --results_root /data_out \
     --skip_spectrograms
   ```

Após completo, todos os arquivos de saída JSON, CSV, e PNG estarão disponíveis no diretório de saída (`/path/to/local/data_out`).

---

## Dados de entrada

Seu dataset (especificado em `--dataset_root`) precisar conter exatamente estes dois diretórios:

1. **input/**

   * Contendo um arquivo `.wav` por sinal de entrada original (ex., `C.wav`, `D.wav`, etc.).

2. **output/**

   * Contendo uma pasta por plugin (ex., `pluginA/`, `pluginB/`).
   * Em cada pasta, deve haver uma pasta por razão de transposição (e.g., `ratio-2/`, `ratio-4/`, `ratio+2/`, `ratio+4/`).
   * Cada pasta de razão de shifting deve conter os arquivos `.wav` trapostos pelo plugin correspondente à pasta (ex., `pluginA/`, `pluginB/`) correspondentes aos sinais originais (com os mesmos nomes, ex. `C.wav`, `D.wav`, etc.).

Exemplo:

```
./MOPAQDAPS/data_in/
├── input/
│   ├── C.wav
│   ├── D.wav
│   └── …  
└── output/
    ├── pluginA/
    │   ├── ratio-2.0/
    │   │   ├── C.wav
    │   │   └── D.wav
    │   └── ratio+2.0/
    │       ├── C.wav
    │       └── D.wav
    ├── pluginB/
    │   ├── ratio-2.0/
    │   │   ├── C.wav
    │   │   └── D.wav
    │   └── ratio+2.0/
    │       ├── C.wav
    │       └── D.wav
    └── …  
```

---

## Dados de saída

Ao executar `mopaqdaps`, serão geradas várias categorias de saída sob o diretório `--results_root`:

### 1. Espectrogramas

**Caminho**: `<RESULTS_ROOT>/spectrograms/`
Cada PNG é nomeado como:

```
<input_basename>_<plugin>_all_ratios.png
```

Exemplo: `C_pluginA_all_ratios.png` mostra  espectrograma do áudio original C.wav ao lado de todos os deus correspondentes transpostos pelo plugin `pluginA` para cada razão de shifting.

### 2. Métricas JSON & CSV

#### `metrics/analytical/`

* Sub-pastas por input (ex. `C/`, `D/`).
* Em cada sub-pasta, um JSON por razão (ex. `ratio-2.json`), contendo:

  ```json
  {
    "input_file": "C.wav",
    "pitch_ratio": "ratio-2.0",
    "original": { … original analytical metrics … },
    "shifted": {
      "pluginA": { … analytical metrics for that shifted file … },
      "pluginB": { … }
    },
    "normalized": {
      "pluginA": { … normalized (shifted / original) per metric … },
      "pluginB": { … }
    }
  }
  ```

#### `metrics/comparison/`

* Mesma estrutura (por input/ratio), mas o JSON inclui:

  ```json
  {
    "input_file": "C.wav",
    "pitch_ratio": "ratio-2",
    "original": { … original comparative metrics (self vs self) … },
    "shifted": {
      "pluginA": { … comparative metrics for (original vs shifted) … },
      "pluginB": { … }
    }
  }
  ```

#### `<RESULTS_ROOT>/results/`

* `original.json`
  Contém um dicinário mapeamndo cada input para suas métricas originais analíticas e comparativas.

* Um JSON por plugin (ex. `pluginA.json`, `pluginB.json`):
  Vetor de registros:

  ```json
  [
    {
      "input_file": "C.wav",
      "pitch_ratio": "ratio-2",
      "analytical": { … },
      "comparison": { … }
    },
    {
      "input_file": "C.wav",
      "pitch_ratio": "ratio-4",
      "analytical": { … },
      "comparison": { … }
    },
    …
  ]
  ```

* `original_metrics.csv`
  Um CSV listando as métricas anlíticas para cada input (uma linha por métrica).

* `original_metrics_zscore.json`
  Resumo de Z-score sobre todas as métricas originais (para ranquear “melhor” vs “pior”).

* `metrics_mean_comparison.csv`
  Tabela de médias por métrica:
  \| metric | original\_mean | pluginA\_mean | pluginB\_mean | … |

* `plugin_composite_ranking.csv`
  Cada linha:
  \| plugin | zscore | avg\_error | std\_error | metric\_count |
  Ordenado por menor zscore (melhor primeiro).

### 3. Gráficos

**CAMINHO**: `<RESULTS_ROOT>/plots/`

* `zscore_by_input_<METRIC>.png`
  Gráfico de linha do z-score por arquivo de entrada, para cada plugin, considerando uma métrica específica.

* `zscore_by_ratio_<METRIC>.png`
  Gráfico de linha do z-score por razão de transposição, para cada plugin, considerando uma métrica específica.

* `zscore_distribution_per_plugin.png`
  Boxplot mostrando a distribuição dos z-scores em todas as métricas para cada plugin.

* `best_plugin_per_input_count.png`
  Gráfico de barras mostrando, para cada arquivo de entrada, qual plugin teve o menor (melhor) z-score; contabiliza quantas vezes cada plugin “venceu”.

* `composite_score_ranking.png`
  Gráfico de barras horizontal ranqueando os plugins pelo seu z-score composto (média dos erros em todas as métricas).


---

## Descrição de Funções

### `analytical_metrics(audio, sr)`

Calcula características de áudio por quadro ou globais para uma única forma de onda:

* **Taxa de Cruzamentos por Zero**: Número de vezes que a forma de onda cruza o zero por quadro (média entre quadros).
* **Energia do Áudio**: Soma dos quadrados das amplitudes dos samples.
* **Medida de Planicidade Espectral**: Planicidade espectral média (tonalidade vs ruído).
* **Tonalidade**: Amplitude absoluta média do componente harmônico (via HPSS).
* **Razão Harmônica**: Aproximada pela média da estimativa de pitch via YIN (em Hz).
* **Centróide Espectral**: Centro de massa no domínio da frequência.
* **Desvio Médio Absoluto de Frequência**: Diferença média absoluta entre centróides espectrais consecutivos.
* **Número de Picos Senoidais**: Contagem de picos espectrais extraídos de um espectrograma.

Retorna um dicionário mapeando o nome de cada métrica ao seu valor calculado.

---

### `comparative_metrics(a1, a2, sr, device=None)`

Calcula características de comparação entre duas formas de onda (`a1` = original, `a2` = transposta):

* **Distância L1 de MFCC**: Distância média L₁ entre vetores MFCC de 20 dimensões.
* **Distância de Sonoridade**: Diferença absoluta média entre valores RMS (√média(|S|²)) por quadro.
* **Desvio de Afinação**: Diferença absoluta média entre estimativas de pitch via PYIN (em Hz).
* **Correlação Cruzada Máxima**: Pico da correlação cruzada total entre os sinais no tempo.
* **Compensação de Desvio**: Diferença absoluta média das fases STFT, escalada pelo número de bins.
* **Distância de Reinício de Fase**: Distância média entre índices onde a fase instantânea "salta" (conta artefatos).

Utiliza PyTorch (e torchaudio) para STFT, MFCC e correlação cruzada. Armazena em cache os transformadores automaticamente se chamado múltiplas vezes com a mesma taxa de amostragem.

---

### `plot_spectrograms(input_dir, output_dir, spectrograms_dir, sr=16000, max_duration=20)`

1. Localiza todos os arquivos `.wav` em `input_dir` e todos os subdiretórios de plugins em `output_dir`.

2. Carrega cada WAV de entrada (truncado para `max_duration` segundos a `sr` de taxa de amostragem), e calcula seu espectrograma em escala logarítmica.

3. Para cada plugin, itera sobre todas as razões de pitch, carrega o áudio transposto (se existir), calcula seu espectrograma e organiza os plots em uma única linha:

   * Coluna 1: espectrograma original.
   * Colunas 2+: espectrogramas de cada razão.

4. Salva cada figura como `<input_basename>_<plugin>_all_ratios.png` em `spectrograms_dir`.

---

### `calculate_metrics(input_dir, output_dir, analytical_dir, comparison_dir, results_dir)`

1. **Métricas Originais**

   * Carrega cada WAV em `input_dir`, calcula `analytical_metrics` e `comparative_metrics(audio, audio)`.
   * Armazena em um dicionário indexado pelo nome do arquivo.

2. **Preparação de Diretórios**

   * Garante que `analytical_dir` e `comparison_dir` tenham subpastas por input (ex.: `analytical/C/`, `comparison/C/`).

3. **Loop por Entrada**
   Para cada WAV de entrada e cada razão de pitch encontrada em qualquer plugin:

   * Cria `analytical_dict`:

     ```json
     {
       "input_file": "C.wav",
       "pitch_ratio": "ratio-2.0",
       "original": { … },
       "shifted": { "pluginA": { … }, "pluginB": { … } },
       "normalized": { "pluginA": { … }, "pluginB": { … } }
     }
     ```

   * Cria `comparison_dict`:

     ```json
     {
       "input_file": "C.wav",
       "pitch_ratio": "ratio-2.0",
       "original": { … },
       "shifted": { "pluginA": { … }, "pluginB": { … } }
     }
     ```

   * Salva como `analytical/<basename>/<ratio>.json` e `comparison/<basename>/<ratio>.json`.

4. **JSON por Plugin**

   * Coleta todos os registros `(input_file, pitch_ratio, analytical, comparison)` por plugin.
   * Escreve um array JSON por plugin em `results_dir`.

5. **CSV de Métricas Originais + Z-Score**

   * Achata todas as métricas analíticas originais em um DataFrame, escreve `original_metrics.csv`.
   * Agrega (média, mediana, desvio padrão, contagem) por métrica, calcula z-scores e salva em `original_metrics_zscore.json`.

6. **Resumo de Médias por Plugin**

   * Para cada plugin, achata os registros (analíticos e comparativos) em um DataFrame, calcula a média por métrica e salva no dicionário `plugin_means`.

7. **CSV de Comparação de Médias**

   * Cria uma tabela no formato `(num_métricas) × (1 + num_plugins)`:
     \| métrica | média\_original | pluginA\_média | pluginB\_média | … |
   * Salva como `metrics_mean_comparison.csv` em `results_dir`.

Retorna dois dicionários:

* `originals_metrics`: mapeia cada nome de arquivo de entrada para suas métricas analíticas e comparativas.
* `plugins_results`: mapeia cada plugin para uma lista de registros `(input_file, pitch_ratio, analytical, comparison)`.

---

### `evaluate_errors(originals_metrics, plugins_results, analytical_dir, comparison_dir, results_dir)`

1. **Coleta de Registros**

   * Itera sobre cada arquivo de entrada, razão de pitch e plugin que possui JSONs analíticos e comparativos.

   * Para cada `(input, ratio, plugin)`, carrega:

     * `analytical_dir/<input_basename>/<ratio>.json`
     * `comparison_dir/<input_basename>/<ratio>.json`

   * Calcula:

     * **Erro analítico** por métrica via `analytical_evaluation(...)`.
     * **Erro comparativo** por métrica via `comparison_evaluation(...)`.

   * Adiciona registros `{ plugin, input, ratio, metric, error }` a uma lista.

2. **CSV Detalhado**

   * Converte a lista de registros em um DataFrame e salva como `results_detailed.csv` em `results_dir`.

3. **Agregações e Z-Scores**
   Para cada agrupamento:

   * **`metric_error_per_plugin`** agrupado por `[plugin, métrica]`

   * **`metric_error_per_plugin_input`** agrupado por `[plugin, input, métrica]`

   * **`metric_error_per_plugin_ratio`** agrupado por `[plugin, ratio, métrica]`

   * **`metric_error_per_plugin_input_ratio`** agrupado por `[plugin, input, ratio, métrica]`

   * Calcula `avg_error`, `median_error`, `std_error`, `metric_count`.

   * Inverte `metric_count` (mais dados = “melhor”).

   * Calcula o Z-Score para `(avg_error, std_error, median_error, metric_count)` e armazena na coluna `zscore`.

   * Salva JSONs aninhados (por níveis de agrupamento) em `results_dir` como `<aggregation_key>.json`.

   * Retém os DataFrames agregados para uso nos gráficos.

4. **Plotagem**
   Em `<RESULTS_ROOT>/plots/`:

   * **Z-score por Entrada**: Um gráfico de linhas por métrica (x=arquivo, y=zscore) por plugin.
   * **Z-score por Razão**: Um gráfico de linhas por métrica (x=razão, y=zscore) por plugin.
   * **Distribuição de Z-score**: Boxplot por plugin.
   * **Melhor Plugin por Entrada**: Gráfico de barras com contagem de vitórias (menor z-score).
   * **Ranking por Z-score Composto**: Gráfico de barras horizontais com ranking dos plugins.
   * Salva cada PNG em `<RESULTS_ROOT>/plots/`.

5. **CSV de Ranking Composto dos Plugins**

   * Agrega média de z-score, média de erro, desvio padrão médio e contagem total de métricas por plugin.
   * Ordena por menor z-score e salva como `plugin_composite_ranking.csv` em `results_dir`.

6. **Valor de Retorno**
   Retorna o DataFrame final `plugin_summary` (indexado por plugin, ordenado por z-score) para que `main()` possa imprimir um ranking textual.

---

## Autores & Contato

* **João Pedro Torres** ([joao.silva@sga.pucminas.br](mailto:joao.silva@sga.pucminas.br))
* **Gabriel Barbosa da Fonseca** ([gabriel.fonseca.771248@sga.pucminas.br](mailto:gabriel.fonseca.771248@sga.pucminas.br))

TCC de Graduação pelo Instituto de Ciências Exatas e Informática

Pontifícia Universidade Católica de Minas Gerais (PUC Minas)

Belo Horizonte, MG, Brasil
