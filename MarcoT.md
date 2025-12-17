# 3. MARCO TEÓRICO

## 3.1 Fundamentos de Seguridad en el Desarrollo de Software

### 3.1.1 Seguridad en el SDLC (Software Development Life Cycle)

La seguridad del software no puede ser un aspecto añadido al final del proceso de desarrollo, sino que debe estar integrada en cada fase del ciclo de vida del desarrollo de software (SDLC). Este enfoque se conoce como **Security by Design** o **Shift-Left Security**.

#### Fases del SDLC Seguro

1. **Planificación**: Identificación de requisitos de seguridad
2. **Diseño**: Modelado de amenazas y arquitectura segura
3. **Implementación**: Codificación segura y revisiones de código
4. **Testing**: Pruebas de seguridad (SAST, DAST, IAST)
5. **Deployment**: Configuración segura y monitoreo
6. **Mantenimiento**: Gestión de vulnerabilidades y parches

**Principio fundamental**: Cuanto antes se detecte una vulnerabilidad, menor es el costo de corrección.

```
Costo de corrección:
- En desarrollo: $100
- En QA: $1,000
- En producción: $10,000+
```

### 3.1.2 Tipos de Vulnerabilidades Comunes

Según el **OWASP Top 10 2021**, las vulnerabilidades más críticas incluyen:

#### A01: Broken Access Control
Fallas en el control de acceso que permiten a usuarios acceder a recursos no autorizados.

#### A02: Cryptographic Failures
Uso inadecuado de criptografía, exposición de datos sensibles.

#### A03: Injection
Inyección de código malicioso en aplicaciones (SQL, Command, LDAP, etc.).

**Ejemplo - SQL Injection:**
```java
// VULNERABLE
String query = "SELECT * FROM users WHERE username='" + userInput + "'";
Statement stmt = conn.createStatement();
ResultSet rs = stmt.executeQuery(query);

// SEGURO
String query = "SELECT * FROM users WHERE username=?";
PreparedStatement stmt = conn.prepareStatement(query);
stmt.setString(1, userInput);
ResultSet rs = stmt.executeQuery();
```

#### A04: Insecure Design
Fallas en el diseño arquitectónico del sistema.

#### A05: Security Misconfiguration
Configuraciones inseguras, permisos excesivos, servicios innecesarios.

#### A06: Vulnerable and Outdated Components
Uso de bibliotecas y frameworks con vulnerabilidades conocidas.

#### A07: Identification and Authentication Failures
Fallas en autenticación, gestión de sesiones, recuperación de contraseñas.

#### A08: Software and Data Integrity Failures
Código y datos no verificados, deserialización insegura.

**Ejemplo - Deserialización Insegura (Python):**
```python
# VULNERABLE
import pickle
data = pickle.loads(untrusted_data)  # Puede ejecutar código arbitrario

# SEGURO
import json
data = json.loads(untrusted_data)  # Solo datos, no código
```

#### A09: Security Logging and Monitoring Failures
Falta de logs, alertas insuficientes, respuesta inadecuada a incidentes.

#### A10: Server-Side Request Forgery (SSRF)
El servidor realiza peticiones a recursos internos manipulados por el atacante.

---

## 3.2 Static Application Security Testing (SAST)

### 3.2.1 Definición y Características

**SAST** (Static Application Security Testing) es una metodología de análisis de seguridad que examina el **código fuente, bytecode o binarios** de una aplicación sin ejecutarla. También conocida como **White-Box Testing**.

#### Características Principales

- **Análisis estático**: No requiere ejecución del código
- **Cobertura completa**: Analiza todo el código fuente
- **Detección temprana**: Identifica vulnerabilidades durante el desarrollo
- **Falsos positivos**: Puede generar alertas incorrectas
- **Lenguaje-específico**: Requiere parsers para cada lenguaje

### 3.2.2 Ventajas y Limitaciones

| Ventajas | Limitaciones |
|----------|--------------|
| ✅ Detección temprana (desarrollo) | ❌ Falsos positivos elevados |
| ✅ Cobertura total del código | ❌ No detecta vulnerabilidades de configuración |
| ✅ Identifica línea exacta del problema | ❌ No detecta vulnerabilidades de lógica compleja |
| ✅ Bajo costo (automatizable) | ❌ Requiere acceso al código fuente |
| ✅ No requiere ambiente de ejecución | ❌ Dificultad con código dinámico |

### 3.2.3 Herramientas SAST Tradicionales

- **SonarQube**: Análisis de calidad y seguridad multi-lenguaje
- **Checkmarx**: Plataforma enterprise de SAST
- **Fortify**: HP Fortify Static Code Analyzer
- **Semgrep**: Análisis de patrones con reglas personalizables
- **Bandit**: Específico para Python
- **ESLint**: JavaScript con plugins de seguridad
- **SpotBugs**: Java (antes FindBugs)

#### Limitación de Reglas Estáticas

Las herramientas SAST tradicionales usan **reglas predefinidas**:

```python
# Regla estática: "Detectar uso de eval()"
if "eval(" in code:
    report_vulnerability("Uso de eval() detectado")
```

**Problemas:**
- No entienden contexto semántico
- Requieren mantenimiento manual de reglas
- Alto ratio de falsos positivos
- No aprenden de nuevos patrones

---

## 3.3 Machine Learning para Detección de Vulnerabilidades

### 3.3.1 ¿Por qué Machine Learning?

El ML ofrece ventajas sobre enfoques basados en reglas:

1. **Aprendizaje automático**: Descubre patrones sin programación explícita
2. **Generalización**: Detecta variantes de vulnerabilidades conocidas
3. **Adaptabilidad**: Se actualiza con nuevos datos
4. **Análisis contextual**: Considera relaciones semánticas en el código
5. **Reducción de falsos positivos**: Mejora con el entrenamiento

### 3.3.2 Enfoque Supervisado

El proyecto utiliza **aprendizaje supervisado**, donde el modelo aprende de ejemplos etiquetados:

```
Datos de Entrenamiento = {(código₁, vulnerable), (código₂, seguro), ...}
                           ↓
                    Algoritmo de ML
                           ↓
                    Modelo Entrenado
                           ↓
          Predicción: nuevo_código → ¿vulnerable?
```

### 3.3.3 Random Forest Classifier

**Random Forest** es un algoritmo de **ensemble learning** que combina múltiples árboles de decisión.

#### Funcionamiento

1. **Bootstrap Aggregating (Bagging)**:
   - Crear N subconjuntos aleatorios del dataset
   - Entrenar un árbol de decisión en cada subconjunto

2. **Feature Randomness**:
   - En cada split, considerar solo un subconjunto aleatorio de features
   - Reduce correlación entre árboles

3. **Voting**:
   - Para clasificación: voto mayoritario
   - Para regresión: promedio

```
Forest = {Árbol₁, Árbol₂, ..., Árbolₙ}

Predicción = vote([Árbol₁(X), Árbol₂(X), ..., Árbolₙ(X)])
```

#### Ventajas para Detección de Vulnerabilidades

- ✅ **Robustez**: Resistente a overfitting
- ✅ **Manejo de datos desbalanceados**: Funciona bien con clases no equiproporcionales
- ✅ **Importancia de features**: Identifica características más relevantes
- ✅ **No linealidad**: Captura relaciones complejas
- ✅ **Paralelizable**: Entrenamiento rápido

#### Hiperparámetros Clave

```python
RandomForestClassifier(
    n_estimators=200,      # Número de árboles
    max_depth=None,        # Profundidad máxima (None = sin límite)
    min_samples_split=2,   # Mínimo de muestras para split
    min_samples_leaf=1,    # Mínimo de muestras en hoja
    max_features='sqrt',   # Features aleatorias por split
    bootstrap=True,        # Usar bootstrap sampling
    random_state=42        # Semilla para reproducibilidad
)
```

---

## 3.4 Feature Engineering para Código Fuente

### 3.4.1 TF-IDF (Term Frequency-Inverse Document Frequency)

**TF-IDF** es una técnica de **vectorización** que convierte texto en números.

#### Fórmula Matemática

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

Donde:
- $\text{TF}(t, d) = \frac{\text{frecuencia de término } t \text{ en documento } d}{\text{total de términos en } d}$
- $\text{IDF}(t) = \log \frac{\text{total de documentos}}{\text{documentos que contienen } t}$

#### Aplicación al Código

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Código como "documento"
codes = [
    "def login(user, password): return authenticate(user, password)",
    "exec(user_input)",
    "SELECT * FROM users WHERE id = ?"
]

vectorizer = TfidfVectorizer(max_features=500)
tfidf_matrix = vectorizer.fit_transform(codes)
# Resultado: matriz sparse de (n_samples, 500)
```

**Ventajas:**
- Captura importancia relativa de tokens
- Reduce peso de palabras comunes (`if`, `for`, `return`)
- Aumenta peso de términos específicos (`exec`, `eval`, `strcpy`)

### 3.4.2 Características Específicas de Seguridad

#### Funciones Peligrosas por Lenguaje

El proyecto define diccionarios de funciones/patrones inseguros:

**Java:**
```python
dangerous_java = [
    "Runtime.getRuntime",  # Ejecución de comandos
    "exec(",                # Command injection
    "Statement",            # SQL injection risk
    "createStatement",
    "ProcessBuilder",       # Process manipulation
    "Class.forName",        # Reflection attacks
    "readObject("           # Deserialization
]
```

**JavaScript:**
```python
dangerous_js = [
    "eval(",                # Code injection
    "innerHTML",            # XSS
    "document.write",       # XSS
    "Function(",            # Dynamic code
    "setTimeout(",          # Code injection risk
    "dangerouslySetInnerHTML"  # React XSS
]
```

**Python:**
```python
dangerous_python = [
    "eval(",                # Code injection
    "exec(",                # Code execution
    "os.system",            # Command injection
    "subprocess.Popen",     # Process execution
    "pickle.loads",         # Deserialization
    "__import__",           # Dynamic imports
    "compile("              # Code compilation
]
```

**C:**
```python
dangerous_c = [
    "strcpy",               # Buffer overflow
    "gets(",                # Buffer overflow
    "scanf(",               # Format string
    "sprintf(",             # Buffer overflow
    "malloc(",              # Memory management
    "system(",              # Command injection
]
```

**C#:**
```python
dangerous_csharp = [
    "Process.Start",        # Process execution
    "SqlCommand",           # SQL injection risk
    "BinaryFormatter",      # Insecure deserialization
    "XmlDocument",          # XXE attacks
    "MD5",                  # Weak cryptography
]
```

#### Conteo de Patrones

```python
def count_dangerous_functions(code, language):
    count = 0
    patterns = dangerous_map.get(language, [])
    for pattern in patterns:
        count += code.count(pattern)
    return count
```

### 3.4.3 Palabras Clave de Vulnerabilidad

```python
vulnerability_keywords = [
    "password", "secret", "token", "api_key",
    "injection", "xss", "csrf", "sql",
    "hardcoded", "plaintext", "unencrypted"
]
```

### 3.4.4 Métricas de Complejidad

```python
def extract_features(code, language):
    features = {
        'code_length': len(code),
        'num_lines': code.count('\n'),
        'dangerous_count': count_dangerous_functions(code, language),
        'density': dangerous_count / max(code_length, 1),
        'has_sql': 1 if 'SELECT' in code or 'INSERT' in code else 0,
        'has_eval': 1 if 'eval(' in code or 'exec(' in code else 0
    }
    return features
```

---

## 3.5 DevSecOps y CI/CD Seguro

### 3.5.1 Definición de DevSecOps

**DevSecOps** = Development + Security + Operations

Es la práctica de **integrar seguridad en cada fase del pipeline DevOps**, automatizando controles de seguridad sin ralentizar el desarrollo.

#### Principios Clave

1. **Shift Left**: Mover seguridad al inicio del SDLC
2. **Automatización**: Pruebas de seguridad automatizadas
3. **Cultura**: Responsabilidad compartida de seguridad
4. **Feedback rápido**: Alertas inmediatas a desarrolladores
5. **Continuous Security**: Monitoreo y mejora continua

### 3.5.2 Pipeline de CI/CD

**CI/CD** (Continuous Integration / Continuous Deployment):

```
Commit → Build → Test → Security Scan → Deploy
          ↓       ↓           ↓            ↓
        Compile  Unit    SAST/DAST    Production
                 Tests   SCA
```

### 3.5.3 GitHub Actions

**GitHub Actions** es la plataforma de CI/CD nativa de GitHub.

#### Componentes Principales

**1. Workflow**: Proceso automatizado definido en YAML

```yaml
name: Security Scan
on: pull_request
jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: python security_check.py
```

**2. Events (Triggers)**:
- `push`: Commit a rama
- `pull_request`: Creación/actualización de PR
- `schedule`: Ejecución periódica
- `workflow_dispatch`: Trigger manual

**3. Jobs**: Conjunto de pasos que se ejecutan en un runner

**4. Steps**: Acciones individuales (checkout, run script, etc.)

**5. Runners**: Máquinas virtuales que ejecutan los workflows
- `ubuntu-latest`
- `windows-latest`
- `macos-latest`

#### Ventajas para DevSecOps

- ✅ Integración nativa con GitHub
- ✅ YAML declarativo y versionado
- ✅ Marketplace de acciones reutilizables
- ✅ Secrets management
- ✅ Matrix builds (múltiples versiones)
- ✅ Artifacts y caching

### 3.5.4 Estrategia de Branching Seguro

```
main ← Producción (protected)
  ↑
test ← Pre-producción (auto-merge si seguro)
  ↑
 dev ← Desarrollo (todos los PRs)
  ↑
feature/* ← Features individuales
```

**Reglas de Protección:**
- `main`: Requiere aprobación + security scan ✅
- `test`: Auto-merge si security scan ✅
- `dev`: Security scan requerido

---

## 3.6 Datasets para Entrenamiento

### 3.6.1 Google CodeXGLUE

**CodeXGLUE** (Code-X General Language Understanding Evaluation) es un benchmark de Microsoft para tareas de comprensión de código.

#### Datasets Utilizados

**1. Code-to-Code Translation**
- Lenguajes: Java ↔ C#
- Tamaño: ~10K pares de traducción
- Uso: Obtener código Java y C# real

**2. Code Completion (Line)**
- Lenguaje: Python
- Tamaño: ~100K ejemplos
- Uso: Código Python de proyectos open-source

**3. Code-to-Text**
- Lenguaje: JavaScript
- Tamaño: ~164K ejemplos
- Uso: Código JavaScript con descripciones

### 3.6.2 BigVul Dataset

**BigVul** es un dataset de vulnerabilidades de código C.

- **Fuente**: Proyectos open-source (Linux Kernel, FFmpeg, etc.)
- **Tamaño**: ~10,000 funciones
- **Etiquetas**: Vulnerable / No vulnerable
- **Tipo de vulnerabilidades**: Buffer overflow, use-after-free, NULL pointer dereference

### 3.6.3 Balanceo de Dataset

```python
# Antes del balanceo
vulnerable: 3,000 (23%)
safe: 10,000 (77%)  ← Desbalanceado

# Después del balanceo (undersampling)
vulnerable: 3,000 (50%)
safe: 3,000 (50%)  ← Balanceado
```

**Técnicas:**
- **Undersampling**: Reducir clase mayoritaria
- **Oversampling**: Aumentar clase minoritaria (SMOTE)
- **Class weighting**: Penalizar más errores en clase minoritaria

---

## 3.7 Métricas de Evaluación de Clasificadores

### 3.7.1 Matriz de Confusión

```
                 Predicción
                SAFE | VULNERABLE
Realidad SAFE    TN  |     FP
         VULN    FN  |     TP
```

Donde:
- **TP** (True Positive): Vulnerable predicho como Vulnerable ✅
- **TN** (True Negative): Seguro predicho como Seguro ✅
- **FP** (False Positive): Seguro predicho como Vulnerable ❌
- **FN** (False Negative): Vulnerable predicho como Seguro ❌❌

### 3.7.2 Métricas Derivadas

#### Accuracy (Exactitud)
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Proporción de predicciones correctas sobre el total.

#### Precision (Precisión)
$$\text{Precision} = \frac{TP}{TP + FP}$$

De los que predijimos como vulnerables, ¿cuántos realmente lo son?
- **Alta precisión**: Pocos falsos positivos (menos alertas innecesarias)

#### Recall (Sensibilidad / Sensitivity)
$$\text{Recall} = \frac{TP}{TP + FN}$$

De los realmente vulnerables, ¿cuántos detectamos?
- **Alto recall**: Pocos falsos negativos (menos vulnerabilidades sin detectar)

#### F1-Score
$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Media armónica de Precision y Recall. Útil cuando necesitamos balance.

### 3.7.3 Trade-offs

```
Aumentar umbral de decisión →
  ↑ Precision (menos FP)
  ↓ Recall (más FN)

Disminuir umbral →
  ↓ Precision (más FP)
  ↑ Recall (menos FN)
```

**Para seguridad**: Preferimos **alto Recall** (detectar todas las vulnerabilidades), tolerando algunos falsos positivos.

---

## 3.8 Sistemas de Notificación

### 3.8.1 Telegram Bot API

**Telegram** ofrece una API robusta para bots:

```python
import requests

def send_telegram_message(token, chat_id, message):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    response = requests.post(url, json=payload)
    return response.json()
```

#### Ventajas
- ✅ API gratuita y sin límites estrictos
- ✅ Soporte para Markdown y HTML
- ✅ Entrega instantánea
- ✅ Historial persistente
- ✅ Multi-plataforma

### 3.8.2 Formato de Mensajes

```markdown
🔒 *SECURITY SCAN RESULT*

Repository: `org/repo`
Pull Request: #123
File: `src/main.py`

Status: ⚠️ *VULNERABLE*
Confidence: 87.5%

Dangerous patterns detected:
• eval() - Code injection risk
• os.system() - Command injection

🔗 [View PR](https://github.com/org/repo/pull/123)
```

---

## 3.9 Conceptos de Seguridad Aplicados

### 3.9.1 Defense in Depth

Múltiples capas de seguridad:

1. **Prevención**: Análisis estático (SAST)
2. **Detección**: Logs y monitoreo
3. **Respuesta**: Alertas y notificaciones
4. **Recuperación**: Rollback automático

### 3.9.2 Principle of Least Privilege

- GitHub tokens con permisos mínimos necesarios
- Secrets en variables de entorno, no en código
- Acceso restrictivo a ramas protegidas

### 3.9.3 Fail-Safe Defaults

- Si el análisis falla → bloquear merge (no aprobar)
- Si hay duda → marcar como vulnerable
- Preferir falsos positivos sobre falsos negativos

---

## 3.10 Referencias Bibliográficas

### Artículos Académicos

1. **DeepVuln**: Deep Learning for Vulnerability Detection
   - Chakraborty, S., et al. (2021)
   - IEEE Symposium on Security and Privacy

2. **VulDeePecker**: Deep Learning-Based Vulnerability Detection
   - Li, Z., et al. (2018)
   - NDSS Symposium

3. **SySeVR**: Vulnerability Detection with Syntax-based Code Representation
   - Li, X., et al. (2019)

### Documentación Técnica

- OWASP Top 10: https://owasp.org/www-project-top-ten/
- CWE (Common Weakness Enumeration): https://cwe.mitre.org/
- GitHub Actions Docs: https://docs.github.com/actions
- scikit-learn Documentation: https://scikit-learn.org/

### Datasets

- CodeXGLUE: https://github.com/microsoft/CodeXGLUE
- BigVul: https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset
- Hugging Face Datasets: https://huggingface.co/datasets

---

**Este marco teórico establece los fundamentos conceptuales y técnicos del proyecto. Para ver cómo se aplicaron estos conceptos, consultar el documento de Metodología.**
