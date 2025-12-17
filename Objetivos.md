# 2. OBJETIVOS

## 2.1 Objetivo General

**Desarrollar e implementar un sistema automatizado de detección de vulnerabilidades de seguridad en código fuente utilizando técnicas de Machine Learning, integrado en un pipeline de CI/CD con GitHub Actions, que permita identificar y clasificar código potencialmente inseguro en múltiples lenguajes de programación (Java, JavaScript, Python, C, C#) antes de su integración en ambientes productivos.**

---

## 2.2 Objetivos Específicos

### 2.2.1 Objetivos de Machine Learning

#### OE1: Construcción del Dataset
**Recopilar y procesar un conjunto de datos diverso y balanceado de código fuente seguro e inseguro en los cinco lenguajes objetivo.**

**Criterios de éxito:**
- ✅ Obtener al menos 2,000 muestras de código por lenguaje
- ✅ Utilizar fuentes confiables (Google CodeXGLUE, BigVul, datasets académicos)
- ✅ Balancear clases (código vulnerable vs seguro) para evitar sesgo
- ✅ Limpiar y estandarizar el formato del código
- ✅ Etiquetar correctamente cada muestra

**Métricas:**
- Número total de muestras: >= 10,000
- Ratio de balanceo: 40-60% entre clases
- Lenguajes cubiertos: Java, C#, C, Python, JavaScript

---

#### OE2: Extracción de Características
**Diseñar e implementar un sistema de extracción de características que capture patrones sintácticos y semánticos relevantes para la detección de vulnerabilidades.**

**Criterios de éxito:**
- ✅ Implementar vectorización TF-IDF para análisis semántico
- ✅ Identificar funciones/patrones peligrosos específicos por lenguaje
- ✅ Calcular métricas de complejidad de código (longitud, densidad)
- ✅ Generar características numéricas y categóricas combinadas

**Características implementadas:**
1. **TF-IDF**: Vectorización del código fuente (max 500 features)
2. **Funciones peligrosas**: Conteo de funciones inseguras por lenguaje
   - Java: `Runtime.getRuntime`, `exec()`, SQL injection patterns
   - JavaScript: `eval()`, `innerHTML`, `document.write`
   - Python: `eval()`, `exec()`, `os.system`, `pickle.loads`
   - C: `strcpy`, `gets()`, buffer overflow patterns
   - C#: `Process.Start`, `SqlCommand`, deserialization insegura
3. **Palabras clave de seguridad**: Detección de términos relacionados con vulnerabilidades
4. **Longitud del código**: Métrica de complejidad
5. **Densidad de funciones peligrosas**: Ratio funciones peligrosas / longitud

---

#### OE3: Entrenamiento del Modelo
**Entrenar y optimizar un modelo de clasificación Random Forest capaz de distinguir código seguro de código vulnerable con alta precisión.**

**Criterios de éxito:**
- ✅ Accuracy >= 85% en conjunto de prueba
- ✅ Precisión >= 80% (minimizar falsos positivos)
- ✅ Recall >= 80% (minimizar falsos negativos)
- ✅ F1-Score >= 80%
- ✅ Implementar validación cruzada

**Configuración del modelo:**
- Algoritmo: Random Forest Classifier
- Número de árboles: 200
- Profundidad máxima: configurable
- Criterio: Gini o Entropy
- Split: train/test 80-20%

---

#### OE4: Persistencia y Deployment del Modelo
**Serializar y almacenar el modelo entrenado junto con sus dependencias para su uso en producción.**

**Criterios de éxito:**
- ✅ Guardar modelo como archivo `.joblib`
- ✅ Guardar vectorizador TF-IDF
- ✅ Guardar columnas de características
- ✅ Documentar versión del modelo
- ✅ Verificar reproducibilidad de predicciones

**Artefactos generados:**
- `ml/model.joblib`: Modelo Random Forest entrenado
- `ml/vectorizer.joblib`: Vectorizador TF-IDF configurado
- `ml/feature_columns.joblib`: Nombres de características

---

### 2.2.2 Objetivos de Ingeniería de Software

#### OE5: Script de Análisis de Seguridad
**Desarrollar un script Python robusto que analice archivos de código y genere reportes de vulnerabilidades.**

**Criterios de éxito:**
- ✅ Detección automática de lenguaje de programación
- ✅ Carga dinámica del modelo ML
- ✅ Extracción de características en tiempo real
- ✅ Clasificación SAFE/VULNERABLE con score de confianza
- ✅ Generación de reporte JSON estructurado
- ✅ Manejo de errores y validaciones

**Funcionalidades del script (`security_check.py`):**
```python
def analyze_code(file_path) -> dict:
    """
    Returns:
    {
        "status": "SAFE" | "VULNERABLE",
        "confidence": 0.0-1.0,
        "file": "path/to/file",
        "language": "python",
        "dangerous_patterns": [...],
        "recommendations": [...]
    }
    """
```

---

#### OE6: Sistema de Notificaciones
**Implementar un bot de Telegram que envíe alertas en tiempo real sobre vulnerabilidades detectadas.**

**Criterios de éxito:**
- ✅ Integración con Telegram Bot API
- ✅ Mensajes formateados con Markdown
- ✅ Información contextual (repositorio, PR, archivo)
- ✅ Enlaces directos a GitHub
- ✅ Manejo de errores de red

**Información enviada:**
- Estado del análisis (✅ SAFE / ⚠️ VULNERABLE)
- Archivo analizado
- Nivel de confianza
- Patrones peligrosos detectados
- Link al Pull Request
- Recomendaciones de seguridad

---

#### OE7: Aplicación Web Frontend
**Crear una interfaz web moderna que presente el proyecto y su funcionalidad.**

**Criterios de éxito:**
- ✅ Diseño responsive y atractivo
- ✅ Información clara sobre características del sistema
- ✅ Despliegue en Vercel
- ✅ Performance optimizado (Next.js)

**Tecnologías:**
- Framework: Next.js 14
- UI: React 18
- Estilos: CSS-in-JS
- Hosting: Vercel

---

### 2.2.3 Objetivos de DevSecOps

#### OE8: Pipeline de CI/CD Seguro
**Diseñar e implementar un pipeline de GitHub Actions que automatice el análisis de seguridad en cada Pull Request.**

**Criterios de éxito:**
- ✅ Trigger automático en PR a dev, test, main
- ✅ Detección de archivos modificados
- ✅ Filtrado de archivos de código (.py, .java, .js, .c, .cs)
- ✅ Instalación de dependencias
- ✅ Ejecución del análisis de seguridad
- ✅ Reporte de resultados en el workflow

**Workflow: `pipeline.yml`**
```yaml
Trigger: pull_request
Branches: [dev, test, main]
Jobs:
  1. Checkout code
  2. Setup Python 3.11
  3. Install dependencies
  4. Detect changed files
  5. Run security check
  6. Report results
```

---

#### OE9: Sistema de Notificación Automatizada
**Integrar el sistema de notificaciones de Telegram en el pipeline CI/CD.**

**Criterios de éxito:**
- ✅ Workflow dedicado para notificaciones
- ✅ Ejecución condicional (después de security scan)
- ✅ Variables de entorno seguras (secrets)
- ✅ Manejo de fallos sin bloquear el pipeline

**Workflow: `notify-telegram.yml`**
- Depende de: security-scan job
- Envía: Resultados del análisis
- Utiliza: TELEGRAM_BOT_TOKEN y CHAT_ID

---

#### OE10: Auto-Merge Inteligente
**Implementar un sistema de merge automático a la rama main basado en el nivel de seguridad del código.**

**Criterios de éxito:**
- ✅ Merge automático solo si código es SAFE
- ✅ Bloqueo de merge si código es VULNERABLE
- ✅ Configuración de permisos adecuados
- ✅ Logs y trazabilidad de decisiones

**Workflow: `auto-merge-to-main.yml`**
```yaml
Conditions:
  - Security status == SAFE
  - All checks passed
  - Target branch == test
Action:
  - Create PR to main
  - Auto-merge
```

---

### 2.2.4 Objetivos de Calidad y Testing

#### OE11: Validación del Modelo
**Evaluar rigurosamente el rendimiento del modelo ML con métricas estándar de clasificación.**

**Métricas objetivo:**
- **Accuracy**: >= 85%
- **Precision**: >= 80% (reducir falsos positivos)
- **Recall**: >= 80% (reducir falsos negativos)
- **F1-Score**: >= 80%
- **ROC-AUC**: >= 0.85
- **Matriz de confusión**: Analizar patrones de error

**Metodología:**
- Train/Test split: 80/20
- Validación cruzada: 5-fold CV
- Análisis de importancia de características
- Curvas de aprendizaje

---

#### OE12: Testing del Pipeline
**Verificar el correcto funcionamiento del pipeline CI/CD con diferentes escenarios.**

**Casos de prueba:**
- ✅ PR con código seguro → SAFE → auto-merge
- ✅ PR con código vulnerable → VULNERABLE → bloqueo
- ✅ PR sin archivos de código → skip
- ✅ Múltiples archivos modificados → análisis correcto
- ✅ Diferentes lenguajes → detección apropiada

---

### 2.2.5 Objetivos de Documentación

#### OE13: Documentación Técnica Completa
**Elaborar documentación exhaustiva que cubra todos los aspectos del proyecto.**

**Documentos requeridos:**
- ✅ Introducción (contexto y problemática)
- ✅ Objetivos (presente documento)
- ✅ Marco Teórico (fundamentos técnicos)
- ✅ Metodología (proceso de desarrollo)
- ✅ Resultados (análisis de performance)
- ✅ Discusión (interpretación y limitaciones)
- ✅ Conclusión (síntesis y trabajos futuros)

**Características de la documentación:**
- Formato Markdown
- Diagramas y visualizaciones
- Ejemplos de código
- Referencias bibliográficas
- Secciones claras y organizadas

---

#### OE14: README del Proyecto
**Crear un README principal que sirva como punto de entrada al proyecto.**

**Contenido:**
- ✅ Descripción general
- ✅ Características principales
- ✅ Arquitectura del sistema
- ✅ Instrucciones de instalación
- ✅ Guía de uso
- ✅ Enlaces a documentación detallada

---

## 2.3 Objetivos Secundarios (Extensiones Futuras)

### OS1: Soporte Multi-vulnerabilidad
Extender el sistema para detectar tipos específicos de vulnerabilidades:
- SQL Injection
- Cross-Site Scripting (XSS)
- Command Injection
- Buffer Overflow
- Insecure Deserialization
- Path Traversal

### OS2: Dashboard de Métricas
Desarrollar un dashboard web que muestre:
- Estadísticas de análisis realizados
- Tendencias de vulnerabilidades detectadas
- Performance del modelo en tiempo real
- Gráficos de evolución temporal

### OS3: Integración con SAST Tools
Combinar el análisis ML con herramientas SAST tradicionales:
- SonarQube
- Semgrep
- Bandit (Python)
- ESLint (JavaScript)

### OS4: API REST
Exponer el modelo como servicio:
- Endpoint `/api/analyze` para análisis on-demand
- Autenticación y rate limiting
- Documentación OpenAPI/Swagger

### OS5: Reentrenamiento Continuo
Implementar pipeline de MLOps:
- Recolección de nuevos datos
- Reentrenamiento periódico
- A/B testing de modelos
- Versionado de modelos

---

## 2.4 Métricas de Éxito del Proyecto

### 2.4.1 Métricas Cuantitativas

| Métrica | Objetivo | Estado |
|---------|----------|--------|
| Accuracy del modelo | >= 85% | ⏳ Por medir |
| F1-Score | >= 80% | ⏳ Por medir |
| Cobertura de lenguajes | 5 lenguajes | ✅ Logrado |
| Tiempo de análisis | < 30 segundos | ⏳ Por medir |
| Uptime del pipeline | >= 95% | ⏳ Por medir |
| Falsos positivos | < 15% | ⏳ Por medir |

### 2.4.2 Métricas Cualitativas

- **Usabilidad**: Sistema fácil de configurar y usar
- **Escalabilidad**: Capacidad de analizar proyectos grandes
- **Mantenibilidad**: Código limpio y bien documentado
- **Seguridad**: Manejo seguro de tokens y credenciales
- **Extensibilidad**: Fácil agregar nuevos lenguajes

---

## 2.5 Alineación con Competencias Académicas

Este proyecto desarrolla competencias clave del curso de **Desarrollo de Software Seguro**:

### Competencias Técnicas
1. ✅ **Secure Coding**: Identificación de patrones inseguros
2. ✅ **DevSecOps**: Integración de seguridad en CI/CD
3. ✅ **Machine Learning**: Aplicación de ML a ciberseguridad
4. ✅ **Automatización**: Pipelines y scripts automatizados
5. ✅ **Testing**: Validación de modelos y sistemas

### Competencias Transversales
1. ✅ **Investigación**: Análisis de datasets y literatura
2. ✅ **Documentación**: Elaboración de documentos técnicos
3. ✅ **Resolución de problemas**: Diseño de soluciones innovadoras
4. ✅ **Gestión de proyectos**: Planificación y ejecución
5. ✅ **Aprendizaje autónomo**: Exploración de nuevas tecnologías

---

## 2.6 Cronograma de Cumplimiento

| Fase | Objetivos | Duración Estimada |
|------|-----------|-------------------|
| **Fase 1: Research** | OE1, OE13 | 1 semana |
| **Fase 2: ML Development** | OE2, OE3, OE4, OE11 | 2 semanas |
| **Fase 3: Scripts** | OE5, OE6 | 1 semana |
| **Fase 4: CI/CD** | OE8, OE9, OE10, OE12 | 1 semana |
| **Fase 5: Frontend** | OE7 | 3 días |
| **Fase 6: Documentation** | OE13, OE14 | 1 semana |
| **Fase 7: Testing & Deploy** | Todos | 3 días |

**Total**: Aproximadamente 6 semanas

---

## 2.7 Indicadores de Logro

Al finalizar el proyecto, se habrá logrado:

✅ **Sistema funcional**: Pipeline CI/CD operativo que analiza PRs automáticamente

✅ **Modelo entrenado**: Clasificador ML con performance >= 85% accuracy

✅ **Integración completa**: GitHub Actions + Telegram + Auto-merge funcionando

✅ **Documentación exhaustiva**: 7 documentos técnicos completos

✅ **Código limpio**: Repositorio organizado con buenas prácticas

✅ **Deployment**: Aplicación web desplegada en Vercel

✅ **Aprendizaje**: Dominio de tecnologías DevSecOps y ML

---

**Nota**: Este documento establece los objetivos del proyecto. El progreso y cumplimiento de cada objetivo se detallará en el documento de Resultados.
