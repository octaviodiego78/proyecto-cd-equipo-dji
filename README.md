# 📊 Proyecto de Análisis y Limpieza de Datos

Este repositorio contiene el trabajo de la *Primera Entrega* del curso, cuyo objetivo es aplicar un flujo completo de análisis exploratorio y limpieza de datos sobre un dataset elegido por el equipo.  

La idea es simular un proyecto real de *ciencia de datos aplicada*, en el que se documenta paso a paso:  
- el entendimiento inicial del dominio,  
- la exploración de los datos disponibles,  
- los problemas encontrados en calidad,  
- las reglas de limpieza aplicadas,  
- y los primeros hallazgos que sirven como base para fases posteriores (modelado, predicción o visualización).  

---

## 📂 Estructura del repositorio



.
├── README.md
├── pyproject.toml # dependencias gestionadas con uv
├── uv.lock # archivo de lock generado automáticamente
├── data/
│ ├── raw/ # datos originales (sin tocar)
│ ├── interim/ # datos intermedios durante la limpieza
│ └── processed/ # dataset final limpio
└── notebooks/
├── 00_informe_final.ipynb # entregable principal del equipo
├── 01_eda_inicial.ipynb # exploración de datos (EDA)
└── 02_data_wrangling.ipynb # reglas de limpieza y normalización


---

## 🧾 Sobre el informe final (00_informe_final.ipynb)

El cuaderno principal contendrá:  
1. *Introducción y contexto*: breve descripción de la industria/dominio elegido y por qué los datos son relevantes.  
2. *Antecedentes*: referencias de trabajos previos o benchmarks en el mismo tema.  
3. *Objetivos*: propósito general y metas específicas que guiarán la entrega.  
4. *Planteamiento del problema*: pregunta de negocio o de análisis a responder con los datos.  
5. *EDA — Resumen narrativo*: hallazgos clave, visualizaciones representativas y riesgos detectados.  
6. *Data Wrangling — Resumen narrativo*: reglas aplicadas para limpiar y transformar datos, con ejemplos antes/después.  
7. *Conclusiones parciales y próximos pasos*: qué se logró y hacia dónde va el proyecto.  
8. *Referencias*: bibliografía o recursos consultados.  

---

## 🔍 Objetivo del proyecto

- *General:* construir un dataset limpio y documentado que pueda ser utilizado en fases posteriores de modelado o visualización.  
- *Específicos:*  
  - Evaluar la calidad de los datos y detectar problemas de consistencia.  
  - Diseñar y aplicar reglas de limpieza reproducibles.  
  - Documentar los hallazgos del EDA con gráficas y métricas clave.  
  - Dejar trazabilidad de cada paso en notebooks ejecutables.  

---

## 🔀 Estrategia de trabajo en Git

- Se trabajará en ramas de features o entregas (ejemplo: feat/entrega-01).  
- Los commits seguirán el estándar [Conventional Commits](https://www.conventionalcommits.org/es/v1.0.0/).  
- Las integraciones a main se harán mediante *Pull Requests*, con al menos una revisión de otro miembro del equipo.  

---

## ✅ Checklist de la entrega

- [ ] Informe en 00_informe_final.ipynb.  
- [ ] 01_eda_inicial.ipynb ejecutado y documentado.  
- [ ] 02_data_wrangling.ipynb ejecutado y documentado.  
- [ ] Dataset limpio en data/processed/.  
- [ ] README con explicación del proyecto y pasos generales.  
- [ ] PR aprobado y mergeado en main.  
- [ ] URL del repositorio entregado en *Canvas*.  

---

## 🧪 Evaluación esperada

| Criterio | Descripción | Peso |
|---|---|---|
| Informe (estructura y claridad) | Introducción, antecedentes, objetivos, problema | 25% |
| EDA | Análisis claro, hallazgos y riesgos identificados | 25% |
| Limpieza | Reglas consistentes, evidencias antes/después | 25% |
| Reproducibilidad & Git | Organización del repo, dataset procesado, commits limpios | 25% |

---

## 🚀 Próximos pasos

A partir de esta base se espera evolucionar hacia:  
- *Feature engineering* (nuevas variables).  
- *Modelos base* para establecer benchmarks.  
- *Visualizaciones y dashboards* que permitan comunicar resultados a un público no técnico.  
- a
---
