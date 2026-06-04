"""
Análisis extendido para responder los 5 puntos de la rúbrica.
Reutiliza la simulación base y agrega:
  - estadísticas agregadas (caja menor/mayor tiempo, promedios por tipo)
  - total de usuarios por tipo por réplica y réplica con menos usuarios
  - análisis M/M/1 teórico (rho, Wq, W) para evaluar si se requiere nueva caja
  - escenario de especialización (cajas exclusivas pagos vs retiros)
"""
import numpy as np
import pandas as pd
import simpy
from simulacion import (TABLA1, TABLA2, P_RETIRO, P_PAGO, JORNADA, N_CAJAS,
                        TIPOS, CATEGORIAS, PROBS, media_entre_llegadas_global,
                        corre_una_replica, resumen_replica)

# ----------------------------------------------------------------------
# 1) Correr 10 réplicas y consolidar
# ----------------------------------------------------------------------
N_REP = 10
todas = []
for r in range(N_REP):
    cajas, _ = corre_una_replica(semilla=1000 + r)
    df = resumen_replica(cajas)
    df.insert(0, "Replica", r + 1)
    todas.append(df)
detalle = pd.concat(todas, ignore_index=True)

prom_caja = detalle.groupby("Caja").mean(numeric_only=True).drop(columns="Replica")

print("="*70)
print("PUNTO 1: Caja con menor y mayor tiempo promedio de atención")
print("="*70)
# 'tiempo de atención' = tiempo en sistema promedio (espera + servicio)
serie = prom_caja["T_sistema_prom"]
print(serie.round(3).to_string())
print(f"\nCaja con MENOR tiempo promedio de atención: Caja {serie.idxmin()} "
      f"({serie.min():.3f} min)")
print(f"Caja con MAYOR tiempo promedio de atención: Caja {serie.idxmax()} "
      f"({serie.max():.3f} min)")
# también por tiempo de servicio puro
serv = prom_caja["T_servicio_prom"]
print(f"\n(Por tiempo de SERVICIO puro -> menor: Caja {serv.idxmin()} "
      f"{serv.min():.3f} min; mayor: Caja {serv.idxmax()} {serv.max():.3f} min)")

print("\n" + "="*70)
print("PUNTO 2: Promedio de usuarios de cada tipo en la totalidad de cajeros")
print("="*70)
# promedio por réplica del total (sumando las 3 cajas), luego promedio
por_rep_tipo = detalle.groupby("Replica")[TIPOS].sum()
prom_tipo_total = por_rep_tipo.mean()
print("Total de usuarios por tipo por réplica (las 3 cajas):")
print(por_rep_tipo.to_string())
print("\nPromedio de usuarios por tipo (en los 3 cajeros):")
print(prom_tipo_total.round(2).to_string())
print(f"\nPromedio total de usuarios/día (3 cajas): "
      f"{detalle.groupby('Replica')['Clientes'].sum().mean():.1f}")

print("\n" + "="*70)
print("PUNTO 3: Total de usuarios por tipo en cada réplica y modelo con menos")
print("="*70)
print(por_rep_tipo.to_string())
total_por_rep = por_rep_tipo.sum(axis=1)
print("\nTotal de usuarios por réplica (todas las cajas):")
print(total_por_rep.to_string())
print(f"\nRéplica con MENOR cantidad de usuarios: réplica {total_por_rep.idxmin()} "
      f"({int(total_por_rep.min())} usuarios)")
# por caja: caja con menos usuarios en promedio
cl = prom_caja["Clientes"]
print(f"Caja con menor cantidad de usuarios (promedio): Caja {cl.idxmin()} "
      f"({cl.min():.1f})")

# ----------------------------------------------------------------------
# PUNTO 4: ¿Se necesita una nueva caja? -> análisis M/M/1 teórico
# ----------------------------------------------------------------------
print("\n" + "="*70)
print("PUNTO 4: ¿Es necesario crear un nuevo cajero? (M/M/1 teórico)")
print("="*70)

# Tasa de llegada total al banco (clientes/min)
lam_total = 1.0 / media_entre_llegadas_global()
# Tiempo medio de servicio global ponderado E[S]
ES = 0.0
for (accion, tipo), p in zip(CATEGORIAS, PROBS):
    ES += p * TABLA1[accion][tipo][0]
mu = 1.0 / ES   # tasa de servicio media (clientes/min)
print(f"Tasa de llegada total al banco  lambda_total = {lam_total:.4f} clientes/min")
print(f"Tiempo medio de servicio global E[S] = {ES:.4f} min  ->  mu = {mu:.4f}/min")

def mm1(lmbda, mu):
    rho = lmbda / mu
    if rho >= 1:
        return rho, np.inf, np.inf, np.inf, np.inf
    Lq = rho**2 / (1 - rho)
    L = rho / (1 - rho)
    Wq = Lq / lmbda
    W = L / lmbda
    return rho, L, Lq, Wq, W

print("\n-- Configuración actual: 3 cajas idénticas, reparto uniforme --")
lam_caja = lam_total / N_CAJAS
rho, L, Lq, Wq, W = mm1(lam_caja, mu)
print(f"  lambda por caja = {lam_caja:.4f}/min")
print(f"  rho = {rho:.4f}  | Lq = {Lq:.3f} | L = {L:.3f}")
print(f"  Wq (espera en cola) = {Wq:.3f} min | W (en sistema) = {W:.3f} min")

print("\n-- Hipótesis: agregar una 4a caja (4 cajas idénticas) --")
lam_caja4 = lam_total / 4
rho4, L4, Lq4, Wq4, W4 = mm1(lam_caja4, mu)
print(f"  lambda por caja = {lam_caja4:.4f}/min")
print(f"  rho = {rho4:.4f} | Wq = {Wq4:.3f} min | W = {W4:.3f} min")

print(f"\n  Wq observado en simulación (prom 3 cajas): "
      f"{prom_caja['T_espera_prom'].mean():.3f} min")

# ----------------------------------------------------------------------
# PUNTO 5: ¿Cuántas cajas exclusivas para pagos y cuántas para retiros?
# Escenario de especialización. Se evalúan reparticiones posibles de 3 cajas.
# ----------------------------------------------------------------------
print("\n" + "="*70)
print("PUNTO 5: Cajas exclusivas para pagos vs retiros (especialización)")
print("="*70)

# tasas y servicio medio por ACCION
def lam_y_ES_por_accion(accion, p_accion):
    lam = 0.0
    es = 0.0
    psum = 0.0
    for tipo in TIPOS:
        p = TABLA2[accion][tipo]
        lam += p_accion * p * (1.0)  # se escala luego con lam_total/sum
        psum += p
    # tasa de llegada de la acción = fracción de lam_total
    frac = p_accion
    lam_accion = lam_total * frac
    # E[S] de la acción
    es = sum(TABLA2[accion][t] * TABLA1[accion][t][0] for t in TIPOS)
    return lam_accion, es

lam_ret, ES_ret = lam_y_ES_por_accion("Retiro", P_RETIRO)
lam_pag, ES_pag = lam_y_ES_por_accion("Consignacion o pago", P_PAGO)
mu_ret = 1.0 / ES_ret
mu_pag = 1.0 / ES_pag
print(f"Retiros : lambda = {lam_ret:.4f}/min, E[S] = {ES_ret:.3f} min, mu = {mu_ret:.4f}")
print(f"Pagos   : lambda = {lam_pag:.4f}/min, E[S] = {ES_pag:.3f} min, mu = {mu_pag:.4f}")

def grupo_mm1(lmbda_total_grupo, mu_grupo, n_cajas):
    """n cajas idénticas que se reparten el flujo de una acción."""
    lam_c = lmbda_total_grupo / n_cajas
    return mm1(lam_c, mu_grupo)

print("\nOpciones de especialización (3 cajas en total):")
opciones = [(1, 2), (2, 1)]   # (retiros, pagos)
res = []
for nret, npag in opciones:
    rho_r, _, Lq_r, Wq_r, W_r = grupo_mm1(lam_ret, mu_ret, nret)
    rho_p, _, Lq_p, Wq_p, W_p = grupo_mm1(lam_pag, mu_pag, npag)
    # espera media ponderada por flujo
    Wq_pond = (lam_ret*Wq_r + lam_pag*Wq_p)/(lam_ret+lam_pag)
    res.append({
        "Retiros(cajas)": nret, "Pagos(cajas)": npag,
        "rho_retiro": rho_r, "Wq_retiro": Wq_r, "W_retiro": W_r,
        "rho_pago": rho_p, "Wq_pago": Wq_p, "W_pago": W_p,
        "Wq_ponderado": Wq_pond,
    })
res_df = pd.DataFrame(res)
print(res_df.round(3).to_string(index=False))

mejor = res_df.loc[res_df["Wq_ponderado"].idxmin()]
print(f"\nMejor reparto por espera ponderada: "
      f"{int(mejor['Retiros(cajas)'])} caja(s) retiros + "
      f"{int(mejor['Pagos(cajas)'])} caja(s) pagos "
      f"(Wq ponderado = {mejor['Wq_ponderado']:.3f} min)")

# Guardar resultados clave
detalle.to_csv("detalle_replicas.csv", index=False)
prom_caja.to_csv("promedio_por_caja.csv")
por_rep_tipo.to_csv("usuarios_por_tipo_replica.csv")
res_df.to_csv("escenarios_especializacion.csv", index=False)

# Guardar un dict de métricas para el reporte
import json
metrics = {
    "media_entre_llegadas": media_entre_llegadas_global(),
    "lam_total": lam_total, "ES": ES, "mu": mu,
    "caja_menor_atencion": int(serie.idxmin()), "t_menor": float(serie.min()),
    "caja_mayor_atencion": int(serie.idxmax()), "t_mayor": float(serie.max()),
    "prom_tipo_total": {k: float(v) for k, v in prom_tipo_total.items()},
    "prom_total_dia": float(detalle.groupby('Replica')['Clientes'].sum().mean()),
    "replica_menos_usuarios": int(total_por_rep.idxmin()),
    "min_usuarios_replica": int(total_por_rep.min()),
    "caja_menos_usuarios": int(cl.idxmin()), "min_clientes_caja": float(cl.min()),
    "actual": {"lam_caja": lam_caja, "rho": rho, "Wq": Wq, "W": W, "Lq": Lq},
    "cuatro_cajas": {"rho": rho4, "Wq": Wq4, "W": W4},
    "Wq_sim": float(prom_caja['T_espera_prom'].mean()),
    "lam_ret": lam_ret, "ES_ret": ES_ret, "mu_ret": mu_ret,
    "lam_pag": lam_pag, "ES_pag": ES_pag, "mu_pag": mu_pag,
    "escenarios": res_df.to_dict(orient="records"),
    "mejor_reparto": {"retiros": int(mejor['Retiros(cajas)']),
                      "pagos": int(mejor['Pagos(cajas)']),
                      "Wq_pond": float(mejor['Wq_ponderado'])},
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print("\nMétricas guardadas en metrics.json")
