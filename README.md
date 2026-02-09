---
jupyter:
  colab:
  kernelspec:
    display_name: .lab1 (3.10.12)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.10.12
  nbformat: 4
  nbformat_minor: 5
---

::: {#ce8ddb55 .cell .markdown id="ce8ddb55"}
# Laboratorio 1: Detección de phishing

-   Pedro Pablo Guzmán Mayen 22111
-   Javier Andres Chen 22153
:::

::: {#ca658de6 .cell .markdown id="ca658de6"}
## Parte 1 - Ingeniería de características

### Exploración de datos

1.  Cargue el dataset en un dataframe de pandas, muestre un ejemplo de
    cinco observaciones.
:::

::: {#beab0587 .cell .code execution_count="44"}
``` python
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from collections import Counter
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score
)

df = pd.read_csv('dataset_pishing.csv')

df.head(5)
```

::: {.output .execute_result execution_count="44"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>http://www.crestonwood.com/router.php</td>
      <td>legitimate</td>
    </tr>
    <tr>
      <th>1</th>
      <td>http://shadetreetechnology.com/V4/validation/a...</td>
      <td>phishing</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://support-appleld.com.secureupdate.duila...</td>
      <td>phishing</td>
    </tr>
    <tr>
      <th>3</th>
      <td>http://rgipt.ac.in</td>
      <td>legitimate</td>
    </tr>
    <tr>
      <th>4</th>
      <td>http://www.iracing.com/tracks/gateway-motorspo...</td>
      <td>legitimate</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#cf5c5d16 .cell .markdown id="cf5c5d16"}
Como vemos, el dataset contiene 2 columnas, status la cuál indica si el
url es legítimo o no y la columna que contiene el url.
:::

::: {#01565530 .cell .markdown id="\"01565530\""}
1.  Muestre la cantidad de observaciones etiquetadas en la columna
    status como "legit" y como "pishing". ¿Está balanceado el dataset?
:::

::: {#ee72999b .cell .code execution_count="2" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":178}" id="ee72999b" outputId="07ba60a0-bc87-40cf-ba98-769a9e9ab7a9"}
``` python
df['status'].value_counts()
```

::: {.output .execute_result execution_count="2"}
    status
    legitimate    5715
    phishing      5715
    Name: count, dtype: int64
:::
:::

::: {#94605174 .cell .markdown id="\"94605174\""}
El dataset si está balanceado pues hay la misma cantidad de urls
maliciosas y no maliciosas.
:::

::: {#ncHmVN-6e0Bj .cell .markdown id="ncHmVN-6e0Bj"}
**Derivacion de caracteristicas**
:::

::: {#ni22JqEEe3FR .cell .markdown id="ni22JqEEe3FR"}
1.  **¿Qué ventajas tiene el análisis de una URL contra el análisis de
    otros datos, cómo el tiempo de vida del dominio, o las
    características de la página Web?**

a\) Detección en tiempo real sin descargar contenido:

-   No requiere acceder al contenido de la página web, lo que evita
    riesgos de seguridad
-   Es significativamente más rápido ya que no necesita descargar HTML,
    CSS, JavaScript o imágenes

b\) Eficiencia computacional:

-   Requiere menos recursos del sistema
-   Permite procesar grandes volúmenes de URLs rápidamente

c\) Independencia de servicios de terceros:

-   Evita retrasos de red causados por servicios externos
-   Funciona incluso cuando servicios de terceros están caídos o son
    lentos

1.  **¿Qué características de una URL son más prometedoras para la
    detección de phishing?**

a\) Características de longitud:

-   Longitud total de la URL
-   Longitud del hostname/dominio

b\) Entropía:

-   Entropía de Shannon de caracteres no-alfanuméricos (propuesta clave)
    Entropía relativa
-   Mide la distribución/desorden de caracteres especiales

c\) Características del dominio:

-   Presencia de dirección IP en lugar de nombre de dominio
-   Posición del TLD (si aparece en path o subdomain es sospechoso)
-   Número de subdominios (phishing usa más)

d\) Características estructurales:

-   Uso de HTTPS (aunque atacantes ahora también lo usan - 78% según
    estudios)
-   Presencia de puertos no estándar
-   Uso de servicios de acortamiento de URLs
:::

::: {#wrZZJVIPiakF .cell .code execution_count="3" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="wrZZJVIPiakF" outputId="9633ce07-66cf-4ebe-d2cd-a7387cbbae73"}
``` python
# FUNCIONES DE EXTRACCIÓN DE CARACTERÍSTICAS

# 1. LONGITUD DE LA URL COMPLETA
def url_length(url):
    return len(url)

# 2. LONGITUD DEL DOMINIO/HOSTNAME
def domain_length(url):
    parsed = urlparse(url)
    return len(parsed.netloc)

# 3. LONGITUD DEL PATH
def path_length(url):
    parsed = urlparse(url)
    return len(parsed.path)

# 4. NÚMERO DE PUNTOS (.)
def count_dots(url):
    return url.count('.')

# 5. NÚMERO DE GUIONES (-)
def count_hyphens(url):
    return url.count('-')

# 6. NÚMERO DE CARACTERES ESPECIALES
def count_special_chars(url):
    special_chars = ['@', '?', '&', '=', '_', '~', '%', '*', '#', '$']
    return sum([url.count(char) for char in special_chars])

# 7. PRESENCIA DE DIRECCIÓN IP
def has_ip_address(url):
    parsed = urlparse(url)
    domain = parsed.netloc
    ip_pattern = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
    return 1 if ip_pattern.search(domain) else 0

# 8. NÚMERO DE SUBDOMINIOS
def count_subdomains(url):
    parsed = urlparse(url)
    domain = parsed.netloc.split(':')[0]  # Remover puerto
    return max(0, domain.count('.') - 1)

# 9. USO DE HTTPS
def is_https(url):
    return 1 if url.startswith('https://') else 0

# 10. NÚMERO DE PARÁMETROS EN LA URL
def count_params(url):
    parsed = urlparse(url)
    query = parsed.query
    if not query:
        return 0
    return len(query.split('&'))

# 11. RATIO DE DÍGITOS EN LA URL
def digit_ratio(url):
    if len(url) == 0:
        return 0
    digits = sum(c.isdigit() for c in url)
    return digits / len(url)

# 12. PRESENCIA DE PALABRAS SENSIBLES
def has_sensitive_words(url):
    keywords = ['login', 'signin', 'account', 'update', 'secure', 'banking',
                'verify', 'confirm', 'password', 'paypal', 'apple']
    url_lower = url.lower()
    return sum(1 for keyword in keywords if keyword in url_lower)

# 13. PROFUNDIDAD DEL PATH
def path_depth(url):
    parsed = urlparse(url)
    path = parsed.path.strip('/')
    if not path:
        return 0
    return len(path.split('/'))

# 14. ENTROPÍA DE SHANNON
def shannon_entropy(url):
    # Extraer solo caracteres no-alfanuméricos
    non_alphanum = [c for c in url if not c.isalnum()]

    if len(non_alphanum) == 0:
        return 0

    # Calcular frecuencias
    freq = Counter(non_alphanum)
    total = len(non_alphanum)

    # Calcular entropía:
    entropy = 0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy

# 15. ENTROPÍA RELATIVA
def relative_entropy(url):
    # Caracteres especiales comunes
    special_chars = ['.', '/', ':', '-', '?', '=', '&', '_', '~', '%']

    # Contar ocurrencias
    counts = {c: url.count(c) for c in special_chars}
    total = sum(counts.values())

    if total == 0:
        return 0

    # Distribución observada
    p = {c: counts[c] / total for c in special_chars}

    # Distribución uniforme
    q = 1 / len(special_chars)

    # Calcular KL divergence:
    kl_div = 0
    for char in special_chars:
        if p[char] > 0:
            kl_div += p[char] * math.log2(p[char] / q)

    return kl_div

# APLICAR TODAS LAS FUNCIONES AL DATASET

print("Extrayendo características de las URLs...")
print("Dataset original shape:", df.shape)

# Aplicar las 15 funciones
df['url_length'] = df['url'].apply(url_length)
df['domain_length'] = df['url'].apply(domain_length)
df['path_length'] = df['url'].apply(path_length)
df['count_dots'] = df['url'].apply(count_dots)
df['count_hyphens'] = df['url'].apply(count_hyphens)
df['count_special_chars'] = df['url'].apply(count_special_chars)
df['has_ip'] = df['url'].apply(has_ip_address)
df['num_subdomains'] = df['url'].apply(count_subdomains)
df['is_https'] = df['url'].apply(is_https)
df['num_params'] = df['url'].apply(count_params)
df['digit_ratio'] = df['url'].apply(digit_ratio)
df['sensitive_word_count'] = df['url'].apply(has_sensitive_words)
df['path_depth'] = df['url'].apply(path_depth)
df['shannon_entropy'] = df['url'].apply(shannon_entropy)
df['relative_entropy'] = df['url'].apply(relative_entropy)

print("\nDataset con características shape:", df.shape)
print("\nPrimeras 5 filas:")
print(df.head())

# Guardar dataset con características
df.to_csv('dataset_with_features.csv', index=False)
print("\n✓ Dataset guardado como 'dataset_with_features.csv'")

# Mostrar estadísticas
print("\n" + "="*80)
print("ESTADÍSTICAS DE LAS 15 CARACTERÍSTICAS")
print("="*80)
print(df.describe())
```

::: {.output .stream .stdout}
    Extrayendo características de las URLs...
    Dataset original shape: (11430, 2)

    Dataset con características shape: (11430, 17)

    Primeras 5 filas:
                                                     url      status  url_length  \
    0              http://www.crestonwood.com/router.php  legitimate          37   
    1  http://shadetreetechnology.com/V4/validation/a...    phishing          77   
    2  https://support-appleld.com.secureupdate.duila...    phishing         126   
    3                                 http://rgipt.ac.in  legitimate          18   
    4  http://www.iracing.com/tracks/gateway-motorspo...  legitimate          55   

       domain_length  path_length  count_dots  count_hyphens  count_special_chars  \
    0             19           11           3              0                    0   
    1             23           47           1              0                    0   
    2             50           20           4              1                    8   
    3             11            0           2              0                    0   
    4             15           33           2              2                    0   

       has_ip  num_subdomains  is_https  num_params  digit_ratio  \
    0       0               1         0           0     0.000000   
    1       0               0         0           0     0.220779   
    2       0               3         1           3     0.150794   
    3       0               1         0           0     0.000000   
    4       0               1         0           0     0.000000   

       sensitive_word_count  path_depth  shannon_entropy  relative_entropy  
    0                     0           1         1.448816          1.873112  
    1                     0           3         1.148835          2.173093  
    2                     3           2         2.755058          0.566870  
    3                     0           0         1.521928          1.800000  
    4                     0           2         1.760964          1.560964  

    ✓ Dataset guardado como 'dataset_with_features.csv'

    ================================================================================
    ESTADÍSTICAS DE LAS 15 CARACTERÍSTICAS
    ================================================================================
             url_length  domain_length   path_length    count_dots  count_hyphens  \
    count  11430.000000   11430.000000  11430.000000  11430.000000   11430.000000   
    mean      61.120035      21.100175     23.146107      2.480665       0.997550   
    std       55.292470      10.778330     27.738075      1.369685       2.087087   
    min       12.000000       4.000000      0.000000      1.000000       0.000000   
    25%       33.000000      15.000000      1.000000      2.000000       0.000000   
    50%       47.000000      19.000000     17.000000      2.000000       0.000000   
    75%       71.000000      24.000000     33.000000      3.000000       1.000000   
    max     1641.000000     214.000000    602.000000     24.000000      43.000000   

           count_special_chars        has_ip  num_subdomains      is_https  \
    count         11430.000000  11430.000000    11430.000000  11430.000000   
    mean              1.078478      0.008486        1.052493      0.389064   
    std               3.302846      0.091734        0.861099      0.487559   
    min               0.000000      0.000000        0.000000      0.000000   
    25%               0.000000      0.000000        1.000000      0.000000   
    50%               0.000000      0.000000        1.000000      0.000000   
    75%               0.000000      0.000000        1.000000      1.000000   
    max              96.000000      1.000000       13.000000      1.000000   

             num_params   digit_ratio  sensitive_word_count    path_depth  \
    count  11430.000000  11430.000000          11430.000000  11430.000000   
    mean       0.291601      0.053141              0.214873      1.830534   
    std        1.024893      0.089367              0.640475      1.887990   
    min        0.000000      0.000000              0.000000      0.000000   
    25%        0.000000      0.000000              0.000000      0.000000   
    50%        0.000000      0.000000              0.000000      2.000000   
    75%        0.000000      0.079365              0.000000      3.000000   
    max       20.000000      0.723881              4.000000     27.000000   

           shannon_entropy  relative_entropy  
    count     11430.000000      11430.000000  
    mean          1.684024          1.653769  
    std           0.386464          0.364514  
    min           0.672312          0.277015  
    25%           1.448816          1.479557  
    50%           1.521928          1.800000  
    75%           1.842371          1.873112  
    max           3.327237          2.649616  
:::
:::

::: {#CUXSk1n8ojb0 .cell .code execution_count="4" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="CUXSk1n8ojb0" outputId="4568c7b0-c7e0-4d29-bf7f-764362f4812b"}
``` python
print("="*80)
print("PREPROCESAMIENTO DEL DATASET")
print("="*80)

# Cargar dataset con características
df = pd.read_csv('dataset_with_features.csv')

print("\n. Estado inicial del dataset:")
print(f"   Shape: {df.shape}")
print(f"   Columnas: {list(df.columns)}")

# CONVERTIR VARIABLE CATEGÓRICA 'status' A BINARIA
print("\n Conversión de variable categórica 'status' a binaria:")
print(f"   Valores únicos antes: {df['status'].unique()}")

# Convertir: phishing = 1, legitimate = 0
df['status'] = df['status'].map({'phishing': 1, 'legitimate': 0})

print(f"   Valores únicos después: {df['status'].unique()}")
print(f"   Distribución:")
print(df['status'].value_counts())


#  ELIMINAR COLUMNA 'url' (DOMINIO/URL ORIGINAL)
print("\n Eliminando columna 'url' ")
print(f"   Columnas antes: {df.shape[1]}")
df = df.drop('url', axis=1)
print(f"   Columnas después: {df.shape[1]}")

# VERIFICAR VALORES NULOS
print("\n Verificación de valores nulos:")
null_counts = df.isnull().sum()
if null_counts.sum() == 0:
    print("    No hay valores nulos en el dataset")
else:
    print("    Valores nulos encontrados:")
    print(null_counts[null_counts > 0])
    # Rellenar con la mediana si hubiera nulos
    df = df.fillna(df.median())
    print("    Valores nulos rellenados con la mediana")

# VERIFICAR VALORES INFINITOS
print("\n Verificación de valores infinitos:")
inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
if inf_counts.sum() == 0:
    print("    No hay valores infinitos en el dataset")
else:
    print("    Valores infinitos encontrados:")
    print(inf_counts[inf_counts > 0])
    # Reemplazar infinitos con NaN y luego con la mediana
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median())
    print("    Valores infinitos reemplazados")

# VERIFICAR DUPLICADOS
print("\n Verificación de filas duplicadas:")
duplicates = df.duplicated().sum()
print(f"   Filas duplicadas encontradas: {duplicates}")

if duplicates > 0:
    df = df.drop_duplicates()
    print(f"    Duplicados eliminados")
    print(f"   Nuevo shape: {df.shape}")
else:
    print("    No hay duplicados")

# VERIFICAR BALANCE DE CLASES
print("\n Balance de clases:")
class_distribution = df['status'].value_counts()
print(class_distribution)
print(f"\n   Proporción:")
print(f"   Legitimate (0): {class_distribution[0]/len(df)*100:.2f}%")
print(f"   Phishing (1): {class_distribution[1]/len(df)*100:.2f}%")

# GUARDAR DATASET PREPROCESADO

# Guardar sin normalizar (para usar después)
df.to_csv('dataset_preprocessed.csv', index=False)
print("\n Dataset preprocesado guardado como 'dataset_preprocessed.csv'")

# NORMALIZACIÓN/ESTANDARIZACIÓN
print("\n Normalización de características (StandardScaler):")

# Separar features y target
X = df.drop('status', axis=1)
y = df['status']

# Aplicar StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Crear DataFrame con datos normalizados
df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
df_scaled['status'] = y.values

print(f"     Características normalizadas (media=0, std=1)")
print(f"\n    Estadísticas después de normalización:")
print(df_scaled.describe())

# Guardar dataset normalizado
df_scaled.to_csv('dataset_preprocessed_scaled.csv', index=False)
print("\n Dataset normalizado guardado como 'dataset_preprocessed_scaled.csv'")
```

::: {.output .stream .stdout}
    ================================================================================
    PREPROCESAMIENTO DEL DATASET
    ================================================================================

    . Estado inicial del dataset:
       Shape: (11430, 17)
       Columnas: ['url', 'status', 'url_length', 'domain_length', 'path_length', 'count_dots', 'count_hyphens', 'count_special_chars', 'has_ip', 'num_subdomains', 'is_https', 'num_params', 'digit_ratio', 'sensitive_word_count', 'path_depth', 'shannon_entropy', 'relative_entropy']

     Conversión de variable categórica 'status' a binaria:
       Valores únicos antes: ['legitimate' 'phishing']
       Valores únicos después: [0 1]
       Distribución:
    status
    0    5715
    1    5715
    Name: count, dtype: int64

     Eliminando columna 'url' 
       Columnas antes: 17
       Columnas después: 16

     Verificación de valores nulos:
        No hay valores nulos en el dataset

     Verificación de valores infinitos:
        No hay valores infinitos en el dataset

     Verificación de filas duplicadas:
       Filas duplicadas encontradas: 3048
        Duplicados eliminados
       Nuevo shape: (8382, 16)

     Balance de clases:
    status
    1    4824
    0    3558
    Name: count, dtype: int64

       Proporción:
       Legitimate (0): 42.45%
       Phishing (1): 57.55%

     Dataset preprocesado guardado como 'dataset_preprocessed.csv'

     Normalización de características (StandardScaler):
         Características normalizadas (media=0, std=1)

        Estadísticas después de normalización:
             url_length  domain_length   path_length    count_dots  count_hyphens  \
    count  8.382000e+03   8.382000e+03  8.382000e+03  8.382000e+03   8.382000e+03   
    mean  -8.137927e-17  -8.582970e-17 -9.324708e-18  1.864942e-17  -2.034482e-17   
    std    1.000060e+00   1.000060e+00  1.000060e+00  1.000060e+00   1.000060e+00   
    min   -9.773944e-01  -1.514040e+00 -9.986509e-01 -1.046762e+00  -5.506357e-01   
    25%   -4.832902e-01  -5.405364e-01 -6.570318e-01 -3.901760e-01  -5.506357e-01   
    50%   -2.447572e-01  -2.750353e-01 -2.129271e-01 -3.901760e-01  -5.506357e-01   
    75%    1.471186e-01   2.559669e-01  3.678253e-01  2.664101e-01   3.016269e-01   
    max    2.677763e+01   1.707104e+01  1.956682e+01  1.405472e+01   1.777301e+01   

           count_special_chars        has_ip  num_subdomains      is_https  \
    count         8.382000e+03  8.382000e+03    8.382000e+03  8.382000e+03   
    mean         -5.065012e-17  7.035916e-17    3.984193e-17 -5.001434e-17   
    std           1.000060e+00  1.000060e+00    1.000060e+00  1.000060e+00   
    min          -3.579293e-01 -1.006127e-01   -1.115509e+00 -7.334723e-01   
    25%          -3.579293e-01 -1.006127e-01   -3.693056e-02 -7.334723e-01   
    50%          -3.579293e-01 -1.006127e-01   -3.693056e-02 -7.334723e-01   
    75%          -7.900002e-02 -1.006127e-01   -3.693056e-02  1.363378e+00   
    max           2.641929e+01  9.939100e+00    1.290601e+01  1.363378e+00   

             num_params   digit_ratio  sensitive_word_count    path_depth  \
    count  8.382000e+03  8.382000e+03          8.382000e+03  8.382000e+03   
    mean  -2.543102e-18  3.443784e-17         -4.280889e-17  5.933905e-17   
    std    1.000060e+00  1.000060e+00          1.000060e+00  1.000060e+00   
    min   -3.089921e-01 -6.577650e-01         -3.666460e-01 -1.202287e+00   
    25%   -3.089921e-01 -6.577650e-01         -3.666460e-01 -6.773590e-01   
    50%   -3.089921e-01 -5.393580e-01         -3.666460e-01 -1.524308e-01   
    75%   -3.089921e-01  3.234475e-01         -3.666460e-01  3.724974e-01   
    max    1.813153e+01  7.013266e+00          6.072803e+00  1.297077e+01   

           shannon_entropy  relative_entropy       status  
    count     8.382000e+03      8.382000e+03  8382.000000  
    mean      4.747124e-17      5.459193e-16     0.575519  
    std       1.000060e+00      1.000060e+00     0.494293  
    min      -2.616844e+00     -3.563484e+00     0.000000  
    25%      -7.927296e-01     -3.789689e-01     0.000000  
    50%      -9.665070e-02      7.195440e-02     1.000000  
    75%       3.564599e-01      7.923005e-01     1.000000  
    max       3.987151e+00      2.740855e+00     1.000000  

     Dataset normalizado guardado como 'dataset_preprocessed_scaled.csv'
:::
:::

::: {#4_CO_Pa2rbrK .cell .markdown id="4_CO_Pa2rbrK"}
**Selección de Características**
:::

::: {#Q5EJLxxjrewp .cell .code execution_count="5" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":1000}" id="Q5EJLxxjrewp" outputId="a6183098-2c2f-4296-9c18-5e48a28b5247"}
``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

sns.set_palette("husl")

print("SELECCIÓN DE CARACTERÍSTICAS")

# CARGAR DATOS
df = pd.read_csv('dataset_preprocessed.csv')
X = df.drop('status', axis=1)
y = df['status']

print(f"\n Dataset original:")
print(f"   Shape: {df.shape}")
print(f"   Características: {X.shape[1]}")


# ANÁLISIS
print("\n" + "=" * 80)
print("ANÁLISIS DE IMPORTANCIA")
print("=" * 80)

# Método 1: Correlación con target
correlations = X.corrwith(y).abs()

# Método 2: ANOVA F-test
f_scores, p_values = f_classif(X, y)

# Método 3: Random Forest Importance
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)
rf_importances = rf.feature_importances_

# Crear DataFrame comparativo
comparison_df = pd.DataFrame({
    'Feature': X.columns,
    'Correlation': correlations.values,
    'F-Score': f_scores,
    'P-Value': p_values,
    'RF_Importance': rf_importances
})

# Normalizar scores (0-1)
for col in ['Correlation', 'F-Score', 'RF_Importance']:
    comparison_df[f'{col}_norm'] = (comparison_df[col] - comparison_df[col].min()) / \
                                    (comparison_df[col].max() - comparison_df[col].min())

# Score promedio
comparison_df['Average_Score'] = comparison_df[
    ['Correlation_norm', 'F-Score_norm', 'RF_Importance_norm']
].mean(axis=1)

comparison_df = comparison_df.sort_values('Average_Score', ascending=False)

print("\nRanking de características:")
print(comparison_df[['Feature', 'Average_Score', 'Correlation', 'P-Value']].to_string())

# VISUALIZACIÓN: COMPARACIÓN DE MÉTODOS
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico 1: Comparación de métodos
x_pos = np.arange(len(comparison_df))
width = 0.25

ax1.bar(x_pos - width, comparison_df['Correlation_norm'], width,
        label='Correlación', alpha=0.8, color='steelblue')
ax1.bar(x_pos, comparison_df['F-Score_norm'], width,
        label='F-Score', alpha=0.8, color='coral')
ax1.bar(x_pos + width, comparison_df['RF_Importance_norm'], width,
        label='RF Importance', alpha=0.8, color='forestgreen')

ax1.set_xlabel('Características', fontsize=11)
ax1.set_ylabel('Score Normalizado', fontsize=11)
ax1.set_title('Comparación de Métodos de Selección', fontsize=13, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(comparison_df['Feature'], rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Gráfico 2: Score promedio
colors = ['darkgreen' if score > 0.5 else 'orange' if score > 0.3 else 'crimson'
          for score in comparison_df['Average_Score']]
ax2.barh(comparison_df['Feature'], comparison_df['Average_Score'], color=colors, alpha=0.7)
ax2.set_xlabel('Score Promedio', fontsize=11)
ax2.set_title('Ranking Final de Características', fontsize=13, fontweight='bold')
ax2.axvline(x=0.3, color='red', linestyle='--', linewidth=2, label='Umbral mínimo')
ax2.axvline(x=0.5, color='orange', linestyle='--', linewidth=2, label='Umbral alto')
ax2.legend()
ax2.grid(axis='x', alpha=0.3)
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('feature_selection_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print(" Gráfico guardado: feature_selection_analysis.png")

# SELECCIÓN DE CARACTERÍSTICAS
print("\n" + "=" * 80)
print("SELECCIÓN FINAL")
print("=" * 80)

# Criterios de selección:
# - Average_Score > 0.3
# - Correlation > 0.05
# - P-Value < 0.05 (estadísticamente significativa)

selected_features = comparison_df[
    (comparison_df['Average_Score'] > 0.3) &
    (comparison_df['Correlation'] > 0.05) &
    (comparison_df['P-Value'] < 0.05)
]['Feature'].tolist()

print(f"\n CARACTERÍSTICAS SELECCIONADAS: {len(selected_features)}/{len(X.columns)}")
print("\nDetalle:")
for i, feature in enumerate(selected_features, 1):
    row = comparison_df[comparison_df['Feature'] == feature].iloc[0]
    print(f"   {i:2d}. {feature:25s} | Score: {row['Average_Score']:.3f} | "
          f"Corr: {row['Correlation']:.3f} | P-val: {row['P-Value']:.2e}")

eliminated = [f for f in X.columns if f not in selected_features]
if eliminated:
    print(f"\n CARACTERÍSTICAS ELIMINADAS: {len(eliminated)}")
    for i, feature in enumerate(eliminated, 1):
        row = comparison_df[comparison_df['Feature'] == feature].iloc[0]
        reasons = []
        if row['Average_Score'] <= 0.3:
            reasons.append(f"Score bajo ({row['Average_Score']:.3f})")
        if row['Correlation'] <= 0.05:
            reasons.append(f"Corr débil ({row['Correlation']:.3f})")
        if row['P-Value'] >= 0.05:
            reasons.append(f"No significativa (p={row['P-Value']:.3f})")
        print(f"   {i}. {feature:25s} | Razón: {', '.join(reasons)}")
else:
    print("\n Todas las características fueron seleccionadas")

# ACTUALIZAR DATASETS EXISTENTES
print("\n" + "=" * 80)
print("ACTUALIZANDO DATASETS")
print("=" * 80)

# Actualizar dataset sin normalizar
df_updated = df[selected_features + ['status']].copy()
df_updated.to_csv('dataset_preprocessed.csv', index=False)
print(f"\n dataset_preprocessed.csv actualizado")
print(f"   Shape anterior: {df.shape}")
print(f"   Shape nuevo: {df_updated.shape}")
print(f"   Características eliminadas: {df.shape[1] - df_updated.shape[1]}")

# Actualizar dataset normalizado
df_scaled = pd.read_csv('dataset_preprocessed_scaled.csv')
df_scaled_updated = df_scaled[selected_features + ['status']].copy()
df_scaled_updated.to_csv('dataset_preprocessed_scaled.csv', index=False)
print(f"\n dataset_preprocessed_scaled.csv actualizado")
print(f"   Shape: {df_scaled_updated.shape}")

```

::: {.output .stream .stdout}
    SELECCIÓN DE CARACTERÍSTICAS

     Dataset original:
       Shape: (8382, 16)
       Características: 15

    ================================================================================
    ANÁLISIS DE IMPORTANCIA
    ================================================================================

    Ranking de características:
                     Feature  Average_Score  Correlation        P-Value
    10           digit_ratio       0.923663     0.280998  6.584431e-152
    11  sensitive_word_count       0.842085     0.288457  2.483673e-160
    1          domain_length       0.674515     0.193609   1.351150e-71
    4          count_hyphens       0.615596     0.217076   5.878602e-90
    0             url_length       0.613869     0.166435   3.975054e-53
    9             num_params       0.460345     0.200381   1.160314e-76
    3             count_dots       0.388383     0.166145   6.033529e-53
    12            path_depth       0.311039     0.126095   4.687366e-31
    2            path_length       0.284232     0.027017   1.337744e-02
    5    count_special_chars       0.252764     0.134231   5.293759e-35
    13       shannon_entropy       0.236749     0.057701   1.248249e-07
    14      relative_entropy       0.226738     0.039966   2.522447e-04
    7         num_subdomains       0.225341     0.078057   8.290247e-13
    6                 has_ip       0.100870     0.086408   2.289782e-15
    8               is_https       0.096323     0.055888   3.056586e-07
:::

::: {.output .display_data}
![](vertopal_ade61a524682410e86543c338eb27d28/0af1a425ef9038d1a15cbd3f606d240edfd802db.png)
:::

::: {.output .stream .stdout}
     Gráfico guardado: feature_selection_analysis.png

    ================================================================================
    SELECCIÓN FINAL
    ================================================================================

     CARACTERÍSTICAS SELECCIONADAS: 8/15

    Detalle:
        1. digit_ratio               | Score: 0.924 | Corr: 0.281 | P-val: 6.58e-152
        2. sensitive_word_count      | Score: 0.842 | Corr: 0.288 | P-val: 2.48e-160
        3. domain_length             | Score: 0.675 | Corr: 0.194 | P-val: 1.35e-71
        4. count_hyphens             | Score: 0.616 | Corr: 0.217 | P-val: 5.88e-90
        5. url_length                | Score: 0.614 | Corr: 0.166 | P-val: 3.98e-53
        6. num_params                | Score: 0.460 | Corr: 0.200 | P-val: 1.16e-76
        7. count_dots                | Score: 0.388 | Corr: 0.166 | P-val: 6.03e-53
        8. path_depth                | Score: 0.311 | Corr: 0.126 | P-val: 4.69e-31

     CARACTERÍSTICAS ELIMINADAS: 7
       1. path_length               | Razón: Score bajo (0.284), Corr débil (0.027)
       2. count_special_chars       | Razón: Score bajo (0.253)
       3. has_ip                    | Razón: Score bajo (0.101)
       4. num_subdomains            | Razón: Score bajo (0.225)
       5. is_https                  | Razón: Score bajo (0.096)
       6. shannon_entropy           | Razón: Score bajo (0.237)
       7. relative_entropy          | Razón: Score bajo (0.227), Corr débil (0.040)

    ================================================================================
    ACTUALIZANDO DATASETS
    ================================================================================

     dataset_preprocessed.csv actualizado
       Shape anterior: (8382, 16)
       Shape nuevo: (8382, 9)
       Características eliminadas: 7

     dataset_preprocessed_scaled.csv actualizado
       Shape: (8382, 9)
:::
:::

::: {#VggD06hCrfgy .cell .markdown id="VggD06hCrfgy"}
1.  **¿Qué columnas o características fueron seleccionadas y por qué?**
:::

::: {#XmECdQAcyB8j .cell .markdown id="XmECdQAcyB8j"}
La selección de características se realizó mediante tres métodos
complementarios: Correlación de Pearson, ANOVA F-test y Random Forest
Feature Importance. Se consideró relevante una característica si cumplía
con los siguientes criterios: score promedio normalizado \> 0.3,
correlación absoluta \> 0.05 y p-value \< 0.05.

De las 15 características originales, se seleccionaron 8 por presentar
el mayor poder discriminativo entre URLs legítimas y de phishing:

-   digit_ratio (score: 0.924, corr: 0.281): fue la característica más
    importante, ya que las URLs de phishing suelen incluir más dígitos
    para generar variantes engañosas de dominios conocidos.

-   sensitive_word_count (score: 0.842, corr: 0.288): detecta palabras
    clave asociadas a instituciones financieras, servicios populares o
    términos de urgencia usados para generar confianza.

-   domain_length (score: 0.675, corr: 0.194): identifica dominios
    excesivamente largos, típicos del phishing que intenta imitar marcas
    legítimas.

-   count_hyphens (score: 0.616, corr: 0.217): captura el uso de
    guiones, una técnica común de typosquatting.

-   url_length (score: 0.614, corr: 0.166): mide la longitud total de la
    URL, generalmente mayor en sitios de phishing debido a subdominios y
    parámetros adicionales.

-   num_params (score: 0.460, corr: 0.200): contabiliza los parámetros
    de la URL, útil para detectar cadenas query sospechosas.

-   count_dots (score: 0.388, corr: 0.166): identifica la presencia de
    múltiples puntos, indicativos de subdominios anidados.

-   path_depth (score: 0.311, corr: 0.126): mide la profundidad de la
    estructura de directorios.

Se descartaron 7 características por baja relevancia estadística o
importancia predictiva: path_length (correlación muy débil),
shannon_entropy y relative_entropy (scores bajos), así como
count_special_chars, num_subdomains, has_ip e is_https, todas por debajo
del umbral establecido.
:::

::: {#ff1612a3 .cell .code execution_count="6"}
``` python
df_scaled_updated.head()
```

::: {.output .execute_result execution_count="6"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>digit_ratio</th>
      <th>sensitive_word_count</th>
      <th>domain_length</th>
      <th>count_hyphens</th>
      <th>url_length</th>
      <th>num_params</th>
      <th>count_dots</th>
      <th>path_depth</th>
      <th>status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.657765</td>
      <td>-0.366646</td>
      <td>-0.186535</td>
      <td>-0.550636</td>
      <td>-0.551443</td>
      <td>-0.308992</td>
      <td>0.266410</td>
      <td>-0.677359</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.681853</td>
      <td>-0.366646</td>
      <td>0.167467</td>
      <td>-0.550636</td>
      <td>0.130080</td>
      <td>-0.308992</td>
      <td>-1.046762</td>
      <td>0.372497</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.940210</td>
      <td>4.462940</td>
      <td>2.556976</td>
      <td>-0.124504</td>
      <td>0.964946</td>
      <td>2.457087</td>
      <td>0.922996</td>
      <td>-0.152431</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.657765</td>
      <td>-0.366646</td>
      <td>-0.894538</td>
      <td>-0.550636</td>
      <td>-0.875166</td>
      <td>-0.308992</td>
      <td>-0.390176</td>
      <td>-1.202287</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.657765</td>
      <td>-0.366646</td>
      <td>-0.540536</td>
      <td>0.301627</td>
      <td>-0.244757</td>
      <td>-0.308992</td>
      <td>-0.390176</td>
      <td>-0.152431</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#22396211 .cell .markdown}
Ya con los datos preprocesados y escalados, vamos a dividir en
validación, prueba y entrenamiento.

Vamos a usar la siguiente división:

-   Entrenamiento: 55%
-   Validación: 15%
-   Prueba: 30%
:::

::: {#b72e2221 .cell .code execution_count="16"}
``` python
df_final = df_scaled_updated

X = df_final.drop("status", axis=1)
y = df_final["status"]

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.2142857,
    random_state=42,
    stratify=y_temp
)

print("Train:", len(X_train) / len(df_final))
print("Validation:", len(X_val) / len(df_final))
print("Test:", len(X_test) / len(df_final))

train_df = X_train.copy()
train_df["status"] = y_train.values
train_df.to_csv("train.csv", index=False)

val_df = X_val.copy()
val_df["status"] = y_val.values
val_df.to_csv("validation.csv", index=False)

test_df = X_test.copy()
test_df["status"] = y_test.values
test_df.to_csv("test.csv", index=False)
```

::: {.output .stream .stdout}
    Train: 0.5498687664041995
    Validation: 0.1500835122882367
    Test: 0.30004772130756385
:::
:::

::: {#4261e07d .cell .markdown}
Ya con el dataset dividido, vamos a crear 3 modelos:

-   Un árbol de decisión: simple de entrenar y muchas veces es efectivo,
    vamos a darle una profundidad de 30 y 25 mustras por hoja
-   Un random forest: más complejo de entrenar pero eficaz, vamos a usar
    3500 estimadores
-   KNN: efectivo para relaciones no lineales, vamos a usar 7 vecinos.
:::

::: {#48d734a1 .cell .code execution_count="45"}
``` python
dt = DecisionTreeClassifier(
    max_depth=30,       
    min_samples_leaf=25, 
    random_state=42
)

dt.fit(X_train, y_train)

rf = RandomForestClassifier(
    n_estimators=3500,   
    random_state=42,
    n_jobs=-1         
)

rf.fit(X_train, y_train)


knn = KNeighborsClassifier(
    n_neighbors=7,     
    weights="distance"
)

knn.fit(X_train, y_train)
```

::: {.output .execute_result execution_count="45"}
```{=html}
<style>#sk-container-id-8 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-8 {
  color: var(--sklearn-color-text);
}

#sk-container-id-8 pre {
  padding: 0;
}

#sk-container-id-8 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-8 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-8 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-8 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-8 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-8 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-8 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-8 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-8 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-8 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-8 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-8 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-8 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-8 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-8 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-8 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-8 div.sk-toggleable__content {
  display: none;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-8 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-8 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-8 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-8 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  display: block;
  width: 100%;
  overflow: visible;
}

#sk-container-id-8 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-8 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-8 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-8 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-8 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-8 div.sk-label label.sk-toggleable__label,
#sk-container-id-8 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-8 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-8 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-8 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-8 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-8 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-8 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-8 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-8 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-8 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-8 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-8 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-8 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}

.estimator-table summary {
    padding: .5rem;
    font-family: monospace;
    cursor: pointer;
}

.estimator-table details[open] {
    padding-left: 0.1rem;
    padding-right: 0.1rem;
    padding-bottom: 0.3rem;
}

.estimator-table .parameters-table {
    margin-left: auto !important;
    margin-right: auto !important;
}

.estimator-table .parameters-table tr:nth-child(odd) {
    background-color: #fff;
}

.estimator-table .parameters-table tr:nth-child(even) {
    background-color: #f6f6f6;
}

.estimator-table .parameters-table tr:hover {
    background-color: #e0e0e0;
}

.estimator-table table td {
    border: 1px solid rgba(106, 105, 104, 0.232);
}

.user-set td {
    color:rgb(255, 94, 0);
    text-align: left;
}

.user-set td.value pre {
    color:rgb(255, 94, 0) !important;
    background-color: transparent !important;
}

.default td {
    color: black;
    text-align: left;
}

.user-set td i,
.default td i {
    color: black;
}

.copy-paste-icon {
    background-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NDggNTEyIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIDYuNy4yIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlL2ZyZWUgQ29weXJpZ2h0IDIwMjUgRm9udGljb25zLCBJbmMuLS0+PHBhdGggZD0iTTIwOCAwTDMzMi4xIDBjMTIuNyAwIDI0LjkgNS4xIDMzLjkgMTQuMWw2Ny45IDY3LjljOSA5IDE0LjEgMjEuMiAxNC4xIDMzLjlMNDQ4IDMzNmMwIDI2LjUtMjEuNSA0OC00OCA0OGwtMTkyIDBjLTI2LjUgMC00OC0yMS41LTQ4LTQ4bDAtMjg4YzAtMjYuNSAyMS41LTQ4IDQ4LTQ4ek00OCAxMjhsODAgMCAwIDY0LTY0IDAgMCAyNTYgMTkyIDAgMC0zMiA2NCAwIDAgNDhjMCAyNi41LTIxLjUgNDgtNDggNDhMNDggNTEyYy0yNi41IDAtNDgtMjEuNS00OC00OEwwIDE3NmMwLTI2LjUgMjEuNS00OCA0OC00OHoiLz48L3N2Zz4=);
    background-repeat: no-repeat;
    background-size: 14px 14px;
    background-position: 0;
    display: inline-block;
    width: 14px;
    height: 14px;
    cursor: pointer;
}
</style><body><div id="sk-container-id-8" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>KNeighborsClassifier(n_neighbors=7, weights=&#x27;distance&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" checked><label for="sk-estimator-id-8" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>KNeighborsClassifier</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.7/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">?<span>Documentation for KNeighborsClassifier</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted" data-param-prefix="">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>
                    
        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('n_neighbors',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">n_neighbors&nbsp;</td>
            <td class="value">7</td>
        </tr>
    

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('weights',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">weights&nbsp;</td>
            <td class="value">&#x27;distance&#x27;</td>
        </tr>
    

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('algorithm',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">algorithm&nbsp;</td>
            <td class="value">&#x27;auto&#x27;</td>
        </tr>
    

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('leaf_size',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">leaf_size&nbsp;</td>
            <td class="value">30</td>
        </tr>
    

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('p',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">p&nbsp;</td>
            <td class="value">2</td>
        </tr>
    

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('metric',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">metric&nbsp;</td>
            <td class="value">&#x27;minkowski&#x27;</td>
        </tr>
    

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('metric_params',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">metric_params&nbsp;</td>
            <td class="value">None</td>
        </tr>
    

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('n_jobs',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">n_jobs&nbsp;</td>
            <td class="value">None</td>
        </tr>
    
                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div><script>function copyToClipboard(text, element) {
    // Get the parameter prefix from the closest toggleable content
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const fullParamName = paramPrefix ? `${paramPrefix}${text}` : text;

    const originalStyle = element.style;
    const computedStyle = window.getComputedStyle(element);
    const originalWidth = computedStyle.width;
    const originalHTML = element.innerHTML.replace('Copied!', '');

    navigator.clipboard.writeText(fullParamName)
        .then(() => {
            element.style.width = originalWidth;
            element.style.color = 'green';
            element.innerHTML = "Copied!";

            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        })
        .catch(err => {
            console.error('Failed to copy:', err);
            element.style.color = 'red';
            element.innerHTML = "Failed!";
            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        });
    return false;
}

document.querySelectorAll('.fa-regular.fa-copy').forEach(function(element) {
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const paramName = element.parentElement.nextElementSibling.textContent.trim();
    const fullParamName = paramPrefix ? `${paramPrefix}${paramName}` : paramName;

    element.setAttribute('title', fullParamName);
});
</script></body>
```
:::
:::

::: {#062b25f4 .cell .markdown}
Ahora vamos a analizar las métricas de cada modelo, vamos a empezar por
el árbol de decisión.
:::

::: {#ea8567a7 .cell .code}
``` python

def model_metrics(model, name):
    plt.figure(figsize=(8,6))
    preds = model.predict(X_val)
    proba = model.predict_proba(X_val)[:,1]

    # Métricas
    precision = precision_score(y_val, preds)
    recall = recall_score(y_val, preds)
    auc = roc_auc_score(y_val, proba)

    print("\n", "="*50)
    print(name)
    print("="*50)
    print("Precision:", precision)
    print("Recall:", recall)
    print("AUC:", auc)

    # Matriz de confusión
    cm = confusion_matrix(y_val, preds)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f"Matriz de Confusión – {name}")
    plt.show()

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_val, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curvas ROC – Comparación de Modelos")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


model_metrics(dt, "Decision tree")
```

::: {.output .stream .stdout}

     ==================================================
    Decision tree
    ==================================================
    Precision: 0.7791005291005291
    Recall: 0.81353591160221
    AUC: 0.8544085087011402
:::

::: {.output .display_data}
    <Figure size 800x600 with 0 Axes>
:::

::: {.output .display_data}
![](vertopal_ade61a524682410e86543c338eb27d28/f9d39808ba4b8b31a13ba772633ec2e6af80dc35.png)
:::

::: {.output .display_data}
![](vertopal_ade61a524682410e86543c338eb27d28/58b73bde1c5cc0194a7b1e3993572c93d69f5525.png)
:::
:::

::: {#61f42125 .cell .markdown}
El árbol de decisión obtuvo un buen rendimiento general. Su precisión
fue del 77%, lo que significa que, de todas las URLs que el modelo
clasificó como phishing, el 77% realmente lo eran. Esta es una métrica
importante, ya que indica que el modelo no genera demasiados falsos
positivos. El recall fue del 81%, lo que implica que el modelo logró
identificar el 81% de todos los casos reales de phishing, lo cual es
fundamental en un problema de seguridad informática. Además, el AUC fue
alto, lo que indica que el modelo tiene una buena capacidad para separar
URLs legítimas de URLs maliciosas. En conjunto, estos resultados
muestran que el modelo es funcional y no presenta signos claros de
overfitting o underfitting.
:::

::: {#0b700c81 .cell .code execution_count="47"}
``` python
model_metrics(rf, "Random forest")
```

::: {.output .stream .stdout}

     ==================================================
    Random forest
    ==================================================
    Precision: 0.8032128514056225
    Recall: 0.8287292817679558
    AUC: 0.8736937943592608
:::

::: {.output .display_data}
    <Figure size 800x600 with 0 Axes>
:::

::: {.output .display_data}
![](vertopal_ade61a524682410e86543c338eb27d28/9e23d9b53d3a93203efda2eb983e3fd552f8bb00.png)
:::

::: {.output .display_data}
![](vertopal_ade61a524682410e86543c338eb27d28/f0a07027a974f29e2cddc6057e389a3e3ce93bf3.png)
:::
:::

::: {#9cceaa79 .cell .markdown}
El Random Forest mostró un rendimiento ligeramente superior al del árbol
de decisión. Su precisión fue aproximadamente del 80%, lo que indica una
reducción adicional de falsos positivos. Asimismo, el modelo logró
identificar cerca del 83% de todas las URLs maliciosas, lo que se
refleja en su recall. El valor de AUC también fue mayor, lo que confirma
que este modelo tiene una mejor capacidad de discriminación entre
phishing y URLs legítimas. Esto era esperado, ya que Random Forest
combina múltiples árboles, lo que mejora la estabilidad y reduce el
error.
:::

::: {#40bf6384 .cell .code execution_count="48"}
``` python
model_metrics(knn, "KNN")
```

::: {.output .stream .stdout}

     ==================================================
    KNN
    ==================================================
    Precision: 0.800807537012113
    Recall: 0.8218232044198895
    AUC: 0.846661803960519
:::

::: {.output .display_data}
    <Figure size 800x600 with 0 Axes>
:::

::: {.output .display_data}
![](vertopal_ade61a524682410e86543c338eb27d28/a000dc30275299447560702346bca3ebe359b7a5.png)
:::

::: {.output .display_data}
![](vertopal_ade61a524682410e86543c338eb27d28/2ee5738ddfb0c86b2d649e56f3b7ab47dacb6305.png)
:::
:::

::: {#2eaaa071 .cell .markdown}
Finalmente, el modelo K-Nearest Neighbors (KNN) obtuvo un rendimiento
comparable a los modelos anteriores. Sus métricas de precisión y recall
fueron similares, aunque el valor de AUC fue ligeramente menor que el de
lo demás. Esto indica que, aunque KNN es capaz de clasificar
correctamente una gran parte de las URLs, su capacidad para separar
ambas clases no es tan robusta como la de los otros modelos.
:::

::: {#f511b2f3 .cell .markdown}
## Parte 3
:::

::: {#a6381715 .cell .markdown}
1.  ¿Cuál es el impacto de clasificar un sitio legítimo como phishing?

Puede tener un impacto negativo pues puede ser que urls de páginas
legítimas de las cuáles somos consumidores sean bloqueadas y nos
perdamos de información importante y no podamos tener acceso a servicios
importantes.

1.  ¿Cuál es el impacto de clasificar un sitio de phishing como
    legítimo?

Clasificar un sitio como phishing incorrectamente puede tener un fuerte
impacto negativo pues urls enviados de forma maliciosa siempre van a
buscar robarse nuestra información tratando de suplantar urls legítimas
por lo que puede ser que alguien no note la diferencia entre el sitio
real y el sitio malcioso y termine entregando información sensible a
agentes maliciosos.

1.  ¿Qué modelo funcionó mejor para la clasificación de phishing? ¿Por
    qué?

El modelo que mejor funciono fue el Random Forest debido a que tuvo gran
precisión, un buen recall y un valor elevado del AUC, lo que indica que
el modelo es capaz de distinguir mejor entre urls maliciosas y no
malciosas. Otra razón por la que puede ser que fue más eficaz que otros
modelos sería por la forma en que funciona el modelo, pues este genera
diferentes datasets con distintas features seleccionadas al azar y usa
esos datasets nuevos para entrenar árboles de decisión y en base a la
decisión final de los árboles creados hace un promedio para tomar una
decisión, se sabe que los árboles de decisión son bastante sensibles a
ligeros cambios en la data original y eso puede generar variación en los
resultados con cambios muy pequeños, sin embargo si hacemos un consenso
general de varios árboles de decisión, cada uno basado en diferentes
características se obtienen mejores resultados y es menos probable que
una ligera variación en la información original cambie de forma
significativa la clasificación y por lo tanto modelo es más capaz de
generalizar la información.

1.  Una empresa desea utilizar su mejor modelo, debido a que sus
    empleados sufren constantes ataques de phishing mediante e-mail. La
    empresa estima que, de un total de 50,000 emails, un 15% son
    phishing. ¿Qué cantidad de alarmas generaría su modelo? ¿Cuántas
    positivas y cuantas negativas? ¿Funciona el modelo para el BR
    propuesto? En caso negativo, ¿qué propone para reducir la cantidad
    de falsas alarmas?

El mejor modelo obtenido fue el Random Forest, el cual presenta
aproximadamente las siguientes métricas:

-   Precisión (Precision) ≈ 0.80\
-   Recall (TPR) ≈ 0.83

Durante el entrenamiento, el conjunto de datos tenía una distribución
balanceada:

  Clase      Cantidad
  ---------- ----------
  Phishing   5,715
  Legítimo   5,715

Es decir, el base rate de phishing en entrenamiento fue 50%.\
Sin embargo, en el escenario real de la empresa, la distribución es
distinta, pues la empresa recibe 50,000 correos electrónicos, de los
cuales el 15% son phishing:

$$
Phishing = 0.15 \times 50,000 = 7,500
$$

$$
Legítimos = 50,000 - 7,500 = 42,500
$$

El recall indica la proporción de phishing reales que el modelo logra
detectar:

$$
Recall = \frac{TP}{TP + FN}
$$

Usando Recall ≈ 0.83:

$$
TP = 0.83 \times 7,500 = 6,225
$$

$$
FN = 7,500 - 6,225 = 1,275
$$

Esto significa que 1,275 correos de phishing pasarían sin ser
detectados.

La precisión se define como:

$$
Precision = \frac{TP}{TP + FP}
$$

Usando Precision ≈ 0.80:

$$
0.80 = \frac{6,225}{6,225 + FP}
$$

Despejando:

$$
FP = 1,556
$$

Esto representa la cantidad de correos legítimos que serían
incorrectamente marcados como phishing.

### Resultados finales

  Tipo de resultado                Cantidad
  -------------------------------- ----------
  Correos phishing reales          7,500
  Correos legítimos                42,500
  **Verdaderos positivos (TP)**    6,225
  **Falsos negativos (FN)**        1,275
  **Falsos positivos (FP)**        1,556
  **Verdaderos negativos (TN)**    40,944
  **Total de alarmas generadas**   7,781

------------------------------------------------------------------------

### Probabilidad real de que una alarma sea phishing

$$
P(Phishing \mid Alarma) = \frac{TP}{TP + FP}
$$

$$
= \frac{6,225}{7,781} \approx 0.80
$$

Esto significa que 1 de cada 5 alertas es falsa, a pesar de que el
modelo tenga un AUC alto.

El modelo fue entrenado con un conjunto balanceado (50% phishing), pero
en la realidad solo el 15% de los correos son phishing.\
Esta diferencia provoca que el modelo genere más falsas alarmas de las
esperadas\*\*, lo que demuestra el efecto de la Base Rate Fallacy:
ignorar la probabilidad real de ocurrencia de un evento lleva a
interpretaciones incorrectas del rendimiento del modelo en producción.

Aunque el Random Forest presenta un excelente desempeño técnico, su uso
directo en un entorno real produciría más de 1,500 falsas alarmas por
cada 50,000 correos, lo que podría afectar la operación de la empresa.\
Por ello, se sugiere calibrar el modelo con probabilidades que respeten
la frecuencia real y así se reduce la cantidad de falsos positivos. Esto
también sería beneficioso pues se ahorra tiempo en entrenar nuevamente
el modelo.
:::
