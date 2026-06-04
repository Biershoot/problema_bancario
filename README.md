# Problema Bancario — Simulación M/M/1 (Banco de Colombia)

Simulación de eventos discretos de las 3 cajas del banco, modeladas como
sistemas M/M/1 independientes, para decidir cómo organizar la atención de
retiros y pagos.

## Contenido
- `Arango_Calderon_Alejandro_problemaBancario.docx` — Informe (introducción, metodología, resultados, conclusiones).
- `Arango_Calderon_Alejandro_problemaBancario.xlsx` — Datos diligenciados: parámetros, detalle por réplica, promedios, usuarios por tipo y análisis de decisión.
- `simulacion.py` — Modelo base M/M/1 (SimPy). Ejecuta 10 réplicas de un día (480 min).
- `analisis.py` — Estadísticas de los 5 puntos + M/M/1 teórico + escenarios de especialización.
- `escenarios_sim.py` — Simulación de cajas especializadas (retiros vs pagos).
- `resultados/` — Evidencias generadas por los scripts (CSV con detalle por réplica, promedios por caja, usuarios por tipo, escenarios y `metrics.json`).

## Requisitos
```
pip install simpy numpy pandas
```

## Ejecución
```
python simulacion.py     # detalle por réplica y caja
python analisis.py       # responde los 5 puntos, genera metrics.json
python escenarios_sim.py # compara configuraciones especializadas
```

## Resultados clave
- ~247 usuarios/día atendidos entre las 3 cajas; utilización ρ ≈ 0.54 (sistema estable).
- No se justifica una 4ª caja (solo reduce la espera ~1.6 min).
- Recomendación: **1 caja exclusiva para retiros y 2 para pagos**, porque los pagos
  tienen un servicio más largo (E[S]=5.2 vs 2.34 min) y necesitan más capacidad.

## Autor
Alejandro Arango Calderón
