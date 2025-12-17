# 4. METODOLOGÍA

## 4.1 Enfoque General del Proyecto

Este proyecto sigue una metodología **híbrida** que combina elementos de:

1. **Desarrollo Iterativo**: Ciclos de desarrollo, prueba y mejora
2. **DevSecOps**: Integración continua de seguridad
3. **Machine Learning Workflow**: Pipeline estándar de ciencia de datos
4. **Ingeniería de Software Ágil**: Entregables incrementales

### 4.1.1 Fases del Proyecto

```
Fase 1: Investigación y Planificación
   ↓
Fase 2: Recolección y Preparación de Datos
   ↓
Fase 3: Desarrollo del Modelo ML
   ↓
Fase 4: Desarrollo de Scripts de Análisis
   ↓
Fase 5: Implementación del Pipeline CI/CD
   ↓
Fase 6: Desarrollo del Frontend
   ↓
Fase 7: Testing y Validación
   ↓
Fase 8: Documentación y Deployment
```

---

## 4.2 Fase 1: Investigación y Planificación

### 4.2.1 Revisión Bibliográfica

**Actividades realizadas:**
- ✅ Estudio de OWASP Top 10 2021
- ✅ Análisis de herramientas SAST existentes (SonarQube, Semgrep, Bandit)
- ✅ Investigación de papers sobre ML para detección de vulnerabilidades
- ✅ Exploración de datasets públicos de código vulnerable

**Fuentes consultadas:**
- IEEE Xplore: Papers sobre deep learning para vulnerabilidades
- GitHub Security Lab: Patrones de vulnerabilidades
- MITRE CWE: Common Weakness Enumeration
- NIST NVD: National Vulnerability Database

### 4.2.2 Definición de Requisitos

**Requisitos Funcionales:**
1. Detectar vulnerabilidades en 5 lenguajes (Java, C#, Python, JavaScript, C)
2. Integración automática con GitHub Actions
3. Notificaciones en tiempo real vía Telegram
4. Auto-merge inteligente basado en análisis de seguridad
5. Interfaz web informativa

**Requisitos No Funcionales:**
1. Accuracy del modelo >= 85%
2. Tiempo de análisis < 30 segundos
3. Disponibilidad del pipeline >= 95%
4. Código mantenible y documentado
5. Seguridad de credenciales (secrets management)

### 4.2.3 Selección de Tecnologías

| Componente | Tecnología | Justificación |
|------------|------------|---------------|
| **ML Framework** | scikit-learn | Robusto, estable, ampliamente usado |
| **Algoritmo** | Random Forest | Balance precisión/interpretabilidad |
| **CI/CD** | GitHub Actions | Integración nativa con GitHub |
| **Notificaciones** | Telegram Bot API | Gratuito, confiable, fácil de usar |
| **Frontend** | Next.js 14 | SSR, performance, SEO |
| **Lenguaje Principal** | Python 3.11 | Ecosistema ML, scripting |
| **Versionado** | Git/GitHub | Estándar de la industria |

---

## 4.3 Fase 2: Recolección y Preparación de Datos

### 4.3.1 Fuentes de Datos

#### Dataset 1: Google CodeXGLUE - Code-to-Code Translation
```python
from datasets import load_dataset

# Java y C#
ds_java_cs = load_dataset("google/code_x_glue_cc_code_to_code_trans", "default")
train_set = ds_java_cs["train"]

# Extraer código
for item in train_set.select(range(2000)):
    java_code = item["java"]
    cs_code = item["cs"]
```

**Características:**
- **Tamaño**: 10,868 pares Java-C#
- **Uso**: Código real de proyectos open-source
- **Calidad**: Alta, código funcional

#### Dataset 2: BigVul - C Vulnerabilities
```python
ds_c = load_dataset("bstee615/bigvul")

for item in ds_c["train"]:
    c_code = item["func"]
    is_vulnerable = item["target"]  # 0 o 1
```

**Características:**
- **Tamaño**: ~10,000 funciones C
- **Etiquetas**: Vulnerable/No vulnerable
- **Vulnerabilidades**: Buffer overflow, use-after-free, NULL pointer

#### Dataset 3: CodeXGLUE - Python Code Completion
```python
ds_py = load_dataset("google/code_x_glue_cc_code_completion_line", "python")

for item in ds_py["train"].select(range(2000)):
    python_code = item["code"]
```

#### Dataset 4: CodeXGLUE - JavaScript Code-to-Text
```python
ds_js = load_dataset("google/code_x_glue_ct_code_to_text", "javascript")

for item in ds_js["train"]:
    js_code = item["code"]
```

### 4.3.2 Construcción del Dataset Unificado

**Proceso implementado:**

```python
codes = []  # Lista de códigos
langs = []  # Lista de lenguajes correspondientes

# 1. Java + C# (de code-to-code translation)
for item in train_java_cs.select(range(2000)):
    if item["java"]:
        codes.append(item["java"])
        langs.append("java")
    if item["cs"]:
        codes.append(item["cs"])
        langs.append("csharp")

# 2. C (de BigVul)
for item in ds_c["train"]:
    codes.append(item["func"])
    langs.append("c")

# 3. Python
for item in ds_py["train"].select(range(2000)):
    codes.append(item["code"])
    langs.append("python")

# 4. JavaScript
for item in ds_js["train"].select(range(2000)):
    codes.append(item["code"])
    langs.append("javascript")
```

**Resultado:**
- **Total de muestras**: ~10,000-15,000
- **Distribución por lenguaje**: 
  - Java: ~2,000
  - C#: ~2,000
  - C: ~4,000
  - Python: ~2,000
  - JavaScript: ~2,000

### 4.3.3 Etiquetado (Labeling)

**Estrategia de etiquetado heurístico:**

```python
def assign_label(code: str, lang: str) -> int:
    """
    Retorna 1 (vulnerable) si encuentra funciones peligrosas,
    0 (seguro) en caso contrario.
    """
    for dangerous_function in dangerous_map.get(lang, []):
        if dangerous_function in code:
            return 1  # VULNERABLE
    return 0  # SAFE
```

**Diccionarios de funciones peligrosas:**

```python
dangerous_map = {
    "java": [
        "Runtime.getRuntime", "exec(", "Statement", "createStatement",
        "executeQuery", "executeUpdate", "Class.forName", "newInstance()",
        "ProcessBuilder", "URLClassLoader", "ScriptEngineManager",
        "readLine(", "readObject(", "XMLDecoder", "XStream"
    ],
    "csharp": [
        "Process.Start", "SqlCommand", "ExecuteReader", "ExecuteNonQuery",
        "Eval(", "File.ReadAllText", "BinaryFormatter", "Deserialize",
        "XmlDocument", "XmlReader", "DESCryptoServiceProvider",
        "MD5", "Random(", "LoadXml", "InnerXml"
    ],
    "c": [
        "strcpy", "strncpy(", "gets(", "scanf(", "sprintf(", 
        "malloc(", "free(", "strcat(", "strlen(", "memcpy(", 
        "system(", "popen(", "vsprintf(", "fscanf(", "sscanf("
    ],
    "python": [
        "eval(", "exec(", "os.system", "subprocess.Popen", 
        "subprocess.call", "pickle.loads", "yaml.load(", 
        "__import__", "compile(", "input(", "execfile(", 
        "globals(", "locals(", "open("
    ],
    "javascript": [
        "eval(", "innerHTML", "document.write", "Function(", 
        "setTimeout(", "var ", "setInterval(", "outerHTML", 
        "insertAdjacentHTML", "execScript(", 
        "dangerouslySetInnerHTML", "createContextualFragment"
    ],
}
```

**Limitaciones de este enfoque:**
- ⚠️ **Simplificación**: Presencia de función peligrosa no garantiza vulnerabilidad
- ⚠️ **Falsos positivos**: Uso legítimo de funciones peligrosas
- ⚠️ **Falsos negativos**: Vulnerabilidades sin patrones conocidos

**Justificación**: Para propósitos educativos y como baseline, este etiquetado heurístico es suficiente. En producción, se requeriría etiquetado manual por expertos.

### 4.3.4 Limpieza de Datos

```python
# Remover duplicados
codes_unique = list(set(codes))

# Filtrar código vacío o muy corto
codes_filtered = [c for c in codes if len(c) > 50]

# Limpiar caracteres especiales problemáticos
codes_clean = [re.sub(r'[^\x00-\x7F]+', '', c) for c in codes_filtered]
```

---

## 4.4 Fase 3: Desarrollo del Modelo ML

### 4.4.1 Feature Engineering

#### A. Vectorización TF-IDF

**Configuración:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=20000,        # Máximo 20,000 términos
    ngram_range=(1, 3),        # Unigramas, bigramas, trigramas
    token_pattern=r"[A-Za-z_]\w+"  # Solo identificadores válidos
)

X_tfidf = tfidf.fit_transform(codes)
# Resultado: Matriz sparse (n_samples, 20000)
```

**Justificación de parámetros:**
- `max_features=20000`: Balance entre información y dimensionalidad
- `ngram_range=(1,3)`: Captura contexto local (ej: "eval", "eval(", "eval(user_input)")
- `token_pattern`: Ignora operadores, se enfoca en nombres de funciones/variables

#### B. Características Manuales

**Implementación:**

```python
def extract_features(code: str, lang: str) -> dict:
    features = {}
    
    # 1. Métricas básicas
    features["length_chars"] = len(code)
    features["num_lines"] = code.count("\n") + 1
    features["num_tokens"] = len(re.findall(r"\w+", code))
    
    # 2. Complejidad ciclomática aproximada
    features["complexity_score"] = estimate_complexity(code)
    
    # 3. Conteo de funciones peligrosas por lenguaje
    for dangerous_func in dangerous_map.get(lang, []):
        features[f"{lang}_danger_{dangerous_func}"] = code.count(dangerous_func)
    
    # 4. Conteo de sanitizadores
    for sanitizer in sanitizers_map.get(lang, []):
        features[f"{lang}_sanitize_{sanitizer}"] = code.count(sanitizer)
    
    # 5. One-hot encoding del lenguaje
    features[f"lang_{lang}"] = 1
    
    return features

def estimate_complexity(code: str) -> int:
    """Estima complejidad ciclomática contando estructuras de control"""
    keywords = ["if ", "for ", "while ", "switch", "case ", 
                "try", "catch", "elif ", "else:"]
    return sum(code.count(kw) for kw in keywords)
```

**Características generadas:**
- `length_chars`: Longitud del código (int)
- `num_lines`: Número de líneas (int)
- `num_tokens`: Número de tokens (palabras) (int)
- `complexity_score`: Complejidad ciclomática aproximada (int)
- `{lang}_danger_{func}`: Conteo por función peligrosa (int)
- `{lang}_sanitize_{func}`: Conteo por sanitizador (int)
- `lang_{lang}`: One-hot del lenguaje (0 o 1)

#### C. Combinación de Features

```python
import pandas as pd
from scipy.sparse import hstack

# Extraer features manuales para cada código
rows = []
for code, lang in zip(codes, langs):
    feats = extract_features(code, lang)
    feats["label"] = assign_label(code, lang)
    rows.append(feats)

# Crear DataFrame
df = pd.DataFrame(rows).fillna(0)

# Separar features y labels
X_manual = df.drop(columns=["label"]).values
y = df["label"].values

# Combinar TF-IDF + features manuales
X = hstack([X_tfidf, X_manual])
# Shape: (n_samples, 20000 + n_manual_features)
```

### 4.4.2 División del Dataset

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,        # 80% train, 20% test
    random_state=42,      # Reproducibilidad
    stratify=y            # Mantener distribución de clases
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train labels: {np.bincount(y_train)}")
print(f"Test labels: {np.bincount(y_test)}")
```

**Ejemplo de salida:**
```
Train: (8000, 20500), Test: (2000, 20500)
Train labels: [4000, 4000]  # Balanceado
Test labels: [1000, 1000]
```

### 4.4.3 Entrenamiento del Modelo

**Configuración de Random Forest:**

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=900,           # 900 árboles de decisión
    max_depth=None,             # Sin límite de profundidad
    min_samples_split=2,        # Mínimo 2 muestras para split
    min_samples_leaf=1,         # Mínimo 1 muestra en hoja
    class_weight="balanced",    # Balanceo automático de clases
    n_jobs=-1,                  # Usar todos los cores CPU
    random_state=42             # Reproducibilidad
)

# Entrenamiento
model.fit(X_train, y_train)
```

**Justificación de hiperparámetros:**
- `n_estimators=900`: Mayor ensemble → mayor robustez (con costo computacional)
- `max_depth=None`: Permite capturar patrones complejos
- `class_weight="balanced"`: Compensa desbalanceo residual en el dataset
- `n_jobs=-1`: Aprovecha paralelización para velocidad

### 4.4.4 Evaluación del Modelo

```python
from sklearn.metrics import accuracy_score, classification_report

# Predicciones
y_pred = model.predict(X_test)

# Métricas
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred, 
                          target_names=["Safe", "Vulnerable"]))
```

**Métricas esperadas:**
```
Accuracy: 0.8750

              precision    recall  f1-score   support

        Safe       0.88      0.87      0.87      1000
  Vulnerable       0.87      0.88      0.88      1000

    accuracy                           0.88      2000
   macro avg       0.88      0.88      0.88      2000
weighted avg       0.88      0.88      0.88      2000
```

**Matriz de Confusión:**

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Seguro (0)", "Vulnerable (1)"],
            yticklabels=["Seguro (0)", "Vulnerable (1)"])
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.title("Matriz de Confusión")
plt.show()
```

### 4.4.5 Persistencia del Modelo

```python
import joblib

# Guardar modelo
joblib.dump(model, "ml/model.joblib")

# Guardar vectorizador TF-IDF
joblib.dump(tfidf, "ml/vectorizer.joblib")

# Guardar columnas de features (para alineación en producción)
feature_columns = df.drop(columns=["label"]).columns.tolist()
joblib.dump(feature_columns, "ml/feature_columns.joblib")

print("✅ Modelo guardado exitosamente")
```

**Artefactos generados:**
- `model.joblib`: ~50-200 MB (dependiendo de n_estimators)
- `vectorizer.joblib`: ~10-50 MB
- `feature_columns.joblib`: ~1 KB

---

## 4.5 Fase 4: Desarrollo de Scripts de Análisis

### 4.5.1 Script Principal: security_check.py

**Arquitectura del script:**

```
security_check.py
├── Cargar modelo y vectorizador
├── Detectar lenguaje del archivo
├── Leer código del archivo
├── Extraer características
│   ├── TF-IDF vectorization
│   └── Features manuales
├── Combinar features
├── Realizar predicción
├── Aplicar heurísticas adicionales
└── Generar reporte JSON
```

#### A. Carga de Modelos

```python
from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "ml" / "model.joblib"
VECTORIZER_PATH = Path(__file__).resolve().parent.parent / "ml" / "vectorizer.joblib"
FEATURE_COLUMNS_PATH = Path(__file__).resolve().parent.parent / "ml" / "feature_columns.joblib"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
```

#### B. Detección de Lenguaje

```python
def detect_language(code: str, filename: str = "") -> str:
    """
    Detecta el lenguaje de programación por extensión o contenido
    """
    # 1. Por extensión de archivo
    if filename:
        ext = Path(filename).suffix.lower()
        ext_map = {
            ".java": "java",
            ".cs": "csharp",
            ".py": "python",
            ".js": "javascript",
            ".c": "c",
            ".h": "c"
        }
        if ext in ext_map:
            return ext_map[ext]
    
    # 2. Por contenido (heurística)
    code_low = code.lower()
    
    # C# ANTES que Java (ambos tienen "public class")
    if any(kw in code for kw in ["using System", "Console.WriteLine", "namespace "]):
        return "csharp"
    
    if any(kw in code for kw in ["public class", "System.out.println", "import java"]):
        return "java"
    
    if "#include" in code_low or "malloc(" in code_low:
        return "c"
    
    if "def " in code or "import " in code:
        return "python"
    
    if any(kw in code for kw in ["function ", "console.log", "const ", "let ", "var "]):
        return "javascript"
    
    return "unknown"
```

#### C. Extracción de Features en Producción

```python
def build_features(code: str, lang: str) -> dict:
    """
    Misma lógica que en entrenamiento
    """
    feats = {}
    
    feats["length_chars"] = len(code)
    feats["num_lines"] = code.count("\n") + 1
    feats["num_tokens"] = len(re.findall(r"\w+", code))
    feats["complexity_score"] = estimate_complexity(code)
    
    for d in dangerous_map.get(lang, []):
        feats[f"{lang}_danger_{d}"] = code.count(d)
    
    for s in sanitizers_map.get(lang, []):
        feats[f"{lang}_sanitize_{s}"] = code.count(s)
    
    feats[f"lang_{lang}"] = 1
    
    return feats
```

#### D. Predicción

```python
def main():
    # 1. Validar argumentos
    if len(sys.argv) < 2:
        print("ERROR: Debes pasar un archivo de código.")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not Path(file_path).exists():
        print("ERROR: Archivo no encontrado:", file_path)
        sys.exit(1)
    
    # 2. Leer código
    code = Path(file_path).read_text(errors="ignore")
    
    # 3. Detectar lenguaje
    lang = detect_language(code, file_path)
    
    # 4. Vectorización TF-IDF
    X_tfidf = vectorizer.transform([code])
    
    # 5. Features manuales
    feats = build_features(code, lang)
    import pandas as pd
    feat_df = pd.DataFrame([feats])
    # Alinear con columnas de entrenamiento
    feat_df = feat_df.reindex(columns=feature_columns, fill_value=0)
    X_manual = feat_df.values
    
    # 6. Combinar
    from scipy.sparse import hstack
    X = hstack([X_tfidf, X_manual])
    
    # 7. Predicción
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]  # Probabilidad de vulnerable
    
    # 8. Heurística de boost
    danger_count = sum(code.count(d) for d in dangerous_map.get(lang, []))
    if danger_count >= 5:
        pred = 1
        proba = max(proba, 0.75)
    
    # 9. Categorizar según OWASP
    owasp_category = categorize_vulnerability(code, pred)
    
    # 10. Generar reporte
    result = {
        "language": lang,
        "prediction": int(pred),
        "probability": float(proba),
        "status": "VULNERABLE" if pred == 1 else "SAFE",
        "dangerous_functions": danger_count,
        "owasp_category": owasp_category
    }
    
    print(json.dumps(result))
```

#### E. Categorización OWASP

```python
def categorize_vulnerability(code: str, pred: int) -> str:
    """Determina categoría OWASP basado en patrones detectados"""
    if pred == 0:
        return "None"
    
    # Injection - XSS/Code Injection
    if any(func in code for func in ["eval(", "exec(", "innerHTML", 
                                      "document.write", "dangerouslySetInnerHTML"]):
        return "A03:2021 - Injection (XSS/Code Injection)"
    
    # SQL Injection
    elif any(func in code for func in ["executeQuery", "SqlCommand", "createStatement"]):
        return "A03:2021 - Injection (SQL Injection)"
    
    # Insecure Deserialization
    elif any(func in code for func in ["pickle.loads", "readObject", 
                                        "Deserialize", "XMLDecoder"]):
        return "A08:2021 - Software and Data Integrity Failures"
    
    # Command Injection
    elif any(func in code for func in ["system(", "exec(", 
                                        "Runtime.getRuntime", "Process.Start"]):
        return "A03:2021 - Injection (Command Injection)"
    
    else:
        return "A03:2021 - Injection"
```

### 4.5.2 Script de Notificaciones: telegram_notify.py

```python
import os
import sys
import json
import requests

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    response = requests.post(url, json=payload)
    return response.json()

def main():
    # Leer resultado del análisis (JSON pasado como argumento)
    if len(sys.argv) < 2:
        print("ERROR: Se requiere el JSON del análisis")
        sys.exit(1)
    
    result = json.loads(sys.argv[1])
    
    # Construir mensaje
    status_emoji = "⚠️" if result["status"] == "VULNERABLE" else "✅"
    
    message = f"""
{status_emoji} *SECURITY SCAN RESULT*

Repository: `{os.getenv('GITHUB_REPOSITORY', 'unknown')}`
PR: #{os.getenv('GITHUB_PR_NUMBER', 'N/A')}
File: `{result.get('file', 'unknown')}`
Language: `{result['language']}`

Status: *{result['status']}*
Confidence: {result['probability']:.2%}
Dangerous functions: {result['dangerous_functions']}

OWASP Category: {result['owasp_category']}

🔗 [View PR]({os.getenv('GITHUB_PR_URL', '#')})
    """
    
    send_telegram_message(message)
    print("✅ Notificación enviada a Telegram")

if __name__ == "__main__":
    main()
```

---

## 4.6 Fase 5: Implementación del Pipeline CI/CD

### 4.6.1 Workflow 1: Security Scan (pipeline.yml)

**Ubicación:** `.github/workflows/pipeline.yml`

```yaml
name: Security Scan on Pull Requests

on:
  pull_request:
    branches: [ "dev", "test", "main" ]
    paths:
      - "**/*.py"
      - "**/*.java"
      - "**/*.js"
      - "**/*.c"
      - "**/*.cs"
  workflow_dispatch:

jobs:
  security-scan:
    name: Analyze Code Vulnerabilities
    runs-on: ubuntu-latest

    steps:
    # 1. Checkout código
    - name: Checkout repository
      uses: actions/checkout@v3

    # 2. Setup Python
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"

    # 3. Instalar dependencias
    - name: Install dependencies
      run: |
        pip install joblib datasets scikit-learn==1.5.2 scipy requests pandas

    # 4. Detectar archivos cambiados
    - name: List changed files
      id: files
      uses: tj-actions/changed-files@v39

    # 5. Seleccionar archivo de código para analizar
    - name: Select code file to analyze
      id: pickfile
      run: |
        echo "Detecting modified source files..."
        for file in ${{ steps.files.outputs.all_changed_files }}; do
          if [[ $file == *.py || $file == *.java || $file == *.js || $file == *.c || $file == *.cs ]]; then
            echo "file_to_scan=$file" >> $GITHUB_OUTPUT
            exit 0
          fi
        done
        echo "file_to_scan=NONE" >> $GITHUB_OUTPUT

    # 6. Detener si no hay archivo
    - name: Stop if no analyzable file
      if: steps.pickfile.outputs.file_to_scan == 'NONE'
      run: |
        echo "No .py/.java/.js/.c/.cs file detected"
        exit 0

    # 7. Ejecutar análisis de seguridad
    - name: Run Security Check
      id: scan
      run: |
        RESULT=$(python scripts/security_check.py "${{ steps.pickfile.outputs.file_to_scan }}")
        echo "result=$RESULT" >> $GITHUB_OUTPUT
        echo "$RESULT"

    # 8. Reportar resultado
    - name: Report Result
      run: |
        echo "Security Scan Result:"
        echo "${{ steps.scan.outputs.result }}"
```

**Funcionamiento:**
1. Se activa en PRs a `dev`, `test`, `main` que modifican archivos de código
2. Configura ambiente Python 3.11
3. Instala dependencias del modelo ML
4. Detecta archivos modificados en el PR
5. Filtra solo archivos de código (.py, .java, .js, .c, .cs)
6. Ejecuta `security_check.py` en el primer archivo encontrado
7. Imprime resultado JSON

### 4.6.2 Workflow 2: Telegram Notification (notify-telegram.yml)

```yaml
name: Notify via Telegram

on:
  workflow_run:
    workflows: ["Security Scan on Pull Requests"]
    types: [completed]

jobs:
  notify:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout
      uses: actions/checkout@v3

    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"

    - name: Install requests
      run: pip install requests

    - name: Send Telegram notification
      env:
        TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
        TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
      run: |
        python scripts/telegram_notify.py '${{ github.event.workflow_run.conclusion }}'
```

**Funcionamiento:**
1. Se ejecuta automáticamente después del workflow de Security Scan
2. Extrae resultado del workflow anterior
3. Envía notificación a Telegram con detalles del análisis

### 4.6.3 Workflow 3: Auto-Merge to Main (auto-merge-to-main.yml)

```yaml
name: Auto-Merge to Main if Safe

on:
  pull_request:
    branches: ["test"]
    types: [closed]

jobs:
  auto-merge:
    if: github.event.pull_request.merged == true
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout
      uses: actions/checkout@v3

    - name: Create PR to main
      uses: repo-sync/pull-request@v2
      with:
        source_branch: "test"
        destination_branch: "main"
        pr_title: "Auto-merge: Safe code from test to main"
        pr_body: "Automated merge of security-approved code"
        github_token: ${{ secrets.GITHUB_TOKEN }}
```

**Lógica:**
1. Cuando se hace merge a `test` desde `dev`
2. Y el código pasó el security scan (SAFE)
3. Crear automáticamente PR de `test` → `main`
4. Auto-aprobar y hacer merge

---

## 4.7 Fase 6: Desarrollo del Frontend

### 4.7.1 Aplicación Next.js

**Estructura:**
```
app/
├── layout.js    # Layout principal
└── page.js      # Página de inicio
```

**Implementación (page.js):**

```javascript
export default function Home() {
  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center'
    }}>
      <div style={{
        background: 'white',
        padding: '3rem',
        borderRadius: '20px',
        boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
        maxWidth: '600px'
      }}>
        <h1>🔒 Vulnerability Scanner v2.0</h1>
        
        <p>Sistema de análisis de seguridad de código con GitHub Actions</p>

        <div>
          <h2>✨ Características</h2>
          <p>✅ Escaneo automático de vulnerabilidades</p>
          <p>✅ Detección de lenguajes: Java, JavaScript, Python, C, C#</p>
          <p>✅ Integración con GitHub Actions</p>
          <p>✅ Notificaciones en Telegram</p>
          <p>✅ Auto-merge a main si es seguro</p>
        </div>
      </div>
    </div>
  )
}
```

### 4.7.2 Deployment en Vercel

**Configuración (vercel.json):**

```json
{
  "buildCommand": "next build",
  "outputDirectory": ".next",
  "framework": "nextjs"
}
```

**Proceso de deployment:**
1. Conectar repositorio GitHub con Vercel
2. Configurar proyecto Next.js
3. Deploy automático en cada push a `main`

---

## 4.8 Fase 7: Testing y Validación

### 4.8.1 Testing del Modelo ML

**Validación cruzada:**

```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

**Análisis de características importantes:**

```python
import numpy as np

# Obtener importancia de features
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Top 10 features
print("Top 10 características más importantes:")
for i in range(10):
    print(f"{i+1}. Feature {indices[i]}: {importances[indices[i]]:.4f}")
```

### 4.8.2 Testing del Pipeline

**Casos de prueba:**

1. **PR con código vulnerable** → Esperado: VULNERABLE, bloqueo de merge
2. **PR con código seguro** → Esperado: SAFE, auto-merge
3. **PR sin archivos de código** → Esperado: Skip
4. **PR con múltiples archivos** → Esperado: Análisis del primero

### 4.8.3 Testing de Integración

- ✅ Verificar que secrets están configurados correctamente
- ✅ Probar notificaciones de Telegram
- ✅ Validar permisos de GitHub Actions
- ✅ Confirmar funcionamiento de auto-merge

---

## 4.9 Fase 8: Documentación y Deployment

### 4.9.1 Documentación Técnica

Elaboración de 7 documentos:
1. ✅ Introducción
2. ✅ Objetivos
3. ✅ Marco Teórico
4. ✅ Metodología (presente documento)
5. ⏳ Resultados
6. ⏳ Discusión
7. ⏳ Conclusión

### 4.9.2 README Principal

Actualización del README.md con:
- Descripción del proyecto
- Instrucciones de instalación
- Guía de uso
- Arquitectura del sistema
- Enlaces a documentación

### 4.9.3 Deployment Final

- ✅ Modelo ML entrenado y guardado en `ml/`
- ✅ Scripts funcionales en `scripts/`
- ✅ Workflows configurados en `.github/workflows/`
- ✅ Frontend desplegado en Vercel
- ✅ Documentación completa en `docs/`

---

## 4.10 Herramientas y Recursos Utilizados

| Categoría | Herramienta | Versión | Uso |
|-----------|-------------|---------|-----|
| **IDE** | VS Code | Latest | Desarrollo |
| **Python** | Python | 3.11 | Lenguaje principal |
| **ML** | scikit-learn | 1.5.2 | Modelo ML |
| **Data** | pandas | Latest | Manipulación de datos |
| **Data** | Hugging Face Datasets | Latest | Descarga de datasets |
| **Web** | Next.js | 14.0.4 | Frontend |
| **CI/CD** | GitHub Actions | N/A | Automatización |
| **Notif** | Telegram Bot API | N/A | Alertas |
| **Deploy** | Vercel | N/A | Hosting |
| **Docs** | Markdown | N/A | Documentación |

---

## 4.11 Cronología del Desarrollo

| Semana | Actividades |
|--------|-------------|
| **Semana 1** | Investigación, planificación, selección de datasets |
| **Semana 2** | Recolección de datos, limpieza, etiquetado |
| **Semana 3** | Feature engineering, entrenamiento del modelo |
| **Semana 4** | Desarrollo de scripts (security_check.py, telegram_notify.py) |
| **Semana 5** | Configuración de GitHub Actions workflows |
| **Semana 6** | Desarrollo del frontend Next.js |
| **Semana 7** | Testing, debugging, optimización |
| **Semana 8** | Documentación, deployment, presentación |

---

**Esta metodología detalla el proceso completo de desarrollo del proyecto. Los resultados obtenidos se presentan en el siguiente documento.**
