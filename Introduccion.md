# 1. INTRODUCCIÓN

## 1.1 Contexto del Proyecto

En la era digital actual, la seguridad del software se ha convertido en un aspecto crítico del desarrollo de aplicaciones. Las vulnerabilidades en el código fuente representan una de las principales amenazas para la integridad, confidencialidad y disponibilidad de los sistemas informáticos. Según el reporte de OWASP (Open Web Application Security Project), las vulnerabilidades en aplicaciones web y de software continúan siendo explotadas por actores maliciosos, causando pérdidas económicas significativas y comprometiendo datos sensibles de millones de usuarios.

El presente proyecto surge como respuesta a la necesidad de automatizar la detección de vulnerabilidades en el código fuente durante las primeras etapas del ciclo de desarrollo de software, específicamente en el proceso de integración continua (CI/CD). La implementación de un sistema de análisis de seguridad automatizado permite identificar patrones de código inseguro antes de que lleguen al ambiente de producción, reduciendo significativamente el riesgo de exposición a ataques cibernéticos.

## 1.2 Problemática

El desarrollo de software moderno enfrenta varios desafíos en materia de seguridad:

### 1.2.1 Detección Manual Insuficiente
La revisión manual de código (code review) realizada por desarrolladores, aunque valiosa, presenta limitaciones significativas:
- **Limitaciones humanas**: Los revisores pueden pasar por alto patrones de vulnerabilidades sutiles o complejas
- **Tiempo y recursos**: El proceso manual es lento y consume recursos valiosos del equipo
- **Inconsistencia**: Diferentes revisores pueden tener diferentes niveles de conocimiento en seguridad
- **Escalabilidad**: A medida que los proyectos crecen, la revisión manual se vuelve cada vez más difícil de gestionar

### 1.2.2 Falta de Estandarización
Muchos equipos de desarrollo carecen de procesos estandarizados para:
- Identificar vulnerabilidades comunes (SQL Injection, XSS, Command Injection, etc.)
- Evaluar el riesgo de seguridad del código antes de hacer merge a producción
- Establecer criterios objetivos de seguridad para aprobar cambios

### 1.2.3 Integración Tardía de Seguridad
Tradicionalmente, las pruebas de seguridad se realizan en etapas tardías del desarrollo:
- Incrementa el costo de corrección de vulnerabilidades
- Retrasa el tiempo de entrega (time-to-market)
- Genera conflictos entre equipos de desarrollo y seguridad

## 1.3 Solución Propuesta

Este proyecto implementa un **sistema automatizado de análisis de vulnerabilidades de código** que utiliza técnicas de Machine Learning para detectar patrones de código inseguro en múltiples lenguajes de programación. La solución se integra directamente en el flujo de trabajo de desarrollo mediante GitHub Actions, proporcionando:

### 1.3.1 Características Principales

**1. Análisis Multi-lenguaje**
- Soporte para Java, JavaScript, Python, C y C#
- Detección de vulnerabilidades específicas de cada lenguaje
- Análisis basado en patrones y características semánticas del código

**2. Modelo de Machine Learning**
- Clasificador Random Forest entrenado con datasets públicos de código vulnerable y seguro
- Extracción de características mediante TF-IDF y análisis de patrones peligrosos
- Capacidad de aprendizaje y mejora continua del modelo

**3. Integración CI/CD**
- Ejecución automática en Pull Requests a las ramas dev, test y main
- Análisis de archivos modificados (.py, .java, .js, .c, .cs)
- Decisión automatizada de merge basada en nivel de riesgo

**4. Sistema de Notificaciones**
- Alertas en tiempo real vía Telegram
- Información detallada sobre vulnerabilidades detectadas
- Recomendaciones automáticas para mitigación

**5. Pipeline Automatizado**
- **Escaneo de seguridad**: Análisis de código con modelo ML
- **Notificación**: Envío de resultados a Telegram
- **Auto-merge inteligente**: Merge automático a main si el código es seguro

## 1.4 Relevancia del Proyecto

### 1.4.1 Impacto Académico
Este proyecto representa la aplicación práctica de conceptos fundamentales de:
- Desarrollo de Software Seguro (Secure Software Development)
- Machine Learning aplicado a la ciberseguridad
- DevSecOps (integración de seguridad en DevOps)
- Automatización de procesos de calidad de software

### 1.4.2 Impacto Profesional
Las competencias desarrolladas son altamente valoradas en la industria:
- Implementación de pipelines CI/CD seguros
- Desarrollo de sistemas de detección de amenazas
- Integración de ML en aplicaciones de seguridad
- Gestión de seguridad en el ciclo de vida del desarrollo

### 1.4.3 Beneficios Tangibles
- **Reducción de costos**: Detección temprana de vulnerabilidades (hasta 100x más barato que corregir en producción)
- **Mejora de calidad**: Código más seguro y robusto
- **Agilidad**: Automatización que no ralentiza el desarrollo
- **Conciencia**: Educación continua del equipo sobre seguridad

## 1.5 Alcance del Documento

Este documento forma parte de una serie de documentación técnica que describe en detalle:

1. **Introducción** (presente documento): Contexto y problemática
2. **Objetivos**: Metas generales y específicas del proyecto
3. **Marco Teórico**: Fundamentos técnicos y conceptuales
4. **Metodología**: Proceso de desarrollo e implementación
5. **Resultados**: Análisis de datos y performance del sistema
6. **Discusión**: Interpretación de resultados y limitaciones
7. **Conclusión**: Síntesis y trabajos futuros

## 1.6 Organización del Proyecto

El proyecto está estructurado en los siguientes componentes:

```
Proyecto2-vulnerability/
├── app/                    # Aplicación Next.js (frontend)
├── ml/                     # Modelo de Machine Learning
│   ├── train.ipynb        # Notebook de entrenamiento
│   ├── model.joblib       # Modelo entrenado
│   ├── vectorizer.joblib  # Vectorizador TF-IDF
│   └── feature_columns.joblib
├── scripts/               # Scripts de análisis
│   ├── security_check.py  # Análisis principal
│   └── telegram_notify.py # Notificaciones
├── .github/workflows/     # GitHub Actions
│   ├── pipeline.yml       # Pipeline principal
│   ├── notify-telegram.yml
│   └── auto-merge-to-main.yml
└── docs/                  # Documentación
```

## 1.7 Tecnologías Utilizadas

### Backend y ML
- **Python 3.11**: Lenguaje principal para ML y scripts
- **scikit-learn 1.5.2**: Framework de Machine Learning
- **joblib**: Persistencia de modelos
- **pandas**: Manipulación de datos
- **Hugging Face Datasets**: Fuentes de datos de entrenamiento

### Frontend
- **Next.js 14**: Framework React para la interfaz web
- **React 18**: Biblioteca UI
- **Vercel**: Plataforma de deployment

### DevOps
- **GitHub Actions**: CI/CD
- **Telegram Bot API**: Sistema de notificaciones
- **Git**: Control de versiones

### Datasets
- **Google CodeXGLUE**: Código Java, C#, JavaScript, Python
- **BigVul**: Vulnerabilidades de código C
- **Datasets personalizados**: Vulnerabilidades conocidas

## 1.8 Justificación Técnica

La elección de Machine Learning sobre reglas estáticas tradicionales (como en herramientas SAST convencionales) se justifica por:

1. **Adaptabilidad**: El modelo puede aprender nuevos patrones sin reprogramación manual
2. **Contexto**: Análisis semántico del código, no solo sintáctico
3. **Reducción de falsos positivos**: Mayor precisión en la clasificación
4. **Escalabilidad**: Procesamiento eficiente de grandes volúmenes de código
5. **Evolución**: Capacidad de reentrenamiento con nuevos datos

## 1.9 Audiencia Objetivo

Este proyecto está dirigido a:

- **Estudiantes** de Desarrollo de Software Seguro
- **Desarrolladores** interesados en DevSecOps
- **Equipos de desarrollo** que buscan automatizar su seguridad
- **Investigadores** en ML aplicado a ciberseguridad
- **Profesionales** de seguridad informática

---

**Nota**: Este documento proporciona una visión general del proyecto. Para detalles técnicos específicos, consultar los documentos de Marco Teórico y Metodología.
