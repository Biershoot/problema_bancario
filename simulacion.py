"""
Simulación del problema bancario - Banco de Colombia
Modelo de 3 cajas independientes tipo M/M/1.

Parámetros (Tabla 1 y Tabla 2):
 - 70% de los usuarios hacen RETIRO, 30% CONSIGNACIÓN/PAGO.
 - Dentro de cada acción, 4 tipos de usuario (Rápido, Normal, Lento, Muy lento)
   con su probabilidad (Tabla 2), su media de tiempo de servicio y su media
   de tiempo entre llegadas (Tabla 1), ambos exponenciales.
 - El banco opera 8 horas/día (480 min). Se simula 1 día y se hacen réplicas.
 - Cada caja se modela como M/M/1 (un servidor, cola FIFO, llegadas y
   servicios exponenciales). Las 3 cajas son independientes.
"""

import simpy
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Datos de las tablas
# ----------------------------------------------------------------------
# Tabla 1: medias (en minutos) de servicio y de tiempo entre llegadas
#   accion -> tipo -> (media_servicio, media_entre_llegadas)
TABLA1 = {
    "Retiro": {
        "Rapido":   (1, 1),
        "Normal":   (2, 2),
        "Lento":    (3, 3),
        "Muy lento":(4, 3),
    },
    "Consignacion o pago": {
        "Rapido":   (3, 1),
        "Normal":   (3, 2),
        "Lento":    (5, 3),
        "Muy lento":(7, 4),
    },
}

# Tabla 2: probabilidad de cada tipo de usuario dentro de su acción
TABLA2 = {
    "Retiro": {
        "Rapido":   0.23,
        "Normal":   0.40,
        "Lento":    0.17,
        "Muy lento":0.20,
    },
    "Consignacion o pago": {
        "Rapido":   0.10,
        "Normal":   0.20,
        "Lento":    0.30,
        "Muy lento":0.40,
    },
}

P_RETIRO = 0.70          # 70% retiros
P_PAGO   = 0.30          # 30% consignaciones/pagos
JORNADA  = 480.0         # 8 horas en minutos
N_CAJAS  = 3
TIPOS    = ["Rapido", "Normal", "Lento", "Muy lento"]


def media_entre_llegadas_global():
    """
    Media global del tiempo entre llegadas combinando acción y tipo.
    Se obtiene como el inverso de la suma de tasas (1/media) ponderada por la
    probabilidad de cada categoría, para construir el flujo agregado de llegadas
    al banco. Cada llegada luego se clasifica por acción y tipo.
    """
    tasa = 0.0
    for accion, p_accion in [("Retiro", P_RETIRO), ("Consignacion o pago", P_PAGO)]:
        for tipo in TIPOS:
            p = p_accion * TABLA2[accion][tipo]
            media_ll = TABLA1[accion][tipo][1]
            tasa += p * (1.0 / media_ll)   # tasa media ponderada
    return 1.0 / tasa


# Distribución conjunta (acción, tipo) para clasificar cada cliente
CATEGORIAS = []
PROBS = []
for accion, p_accion in [("Retiro", P_RETIRO), ("Consignacion o pago", P_PAGO)]:
    for tipo in TIPOS:
        CATEGORIAS.append((accion, tipo))
        PROBS.append(p_accion * TABLA2[accion][tipo])
PROBS = np.array(PROBS)
PROBS = PROBS / PROBS.sum()   # normalizar por seguridad


class Caja:
    """Una caja M/M/1."""
    def __init__(self, env, idx):
        self.env = env
        self.idx = idx
        self.servidor = simpy.Resource(env, capacity=1)
        # registros
        self.n_clientes = 0
        self.tiempos_espera = []      # tiempo en cola (W_q)
        self.tiempos_sistema = []     # tiempo total en sistema (W)
        self.tiempos_servicio = []    # tiempo de servicio
        self.por_tipo = {t: 0 for t in TIPOS}
        self.por_accion = {"Retiro": 0, "Consignacion o pago": 0}


def cliente(env, caja, accion, tipo, rng):
    llegada = env.now
    media_serv = TABLA1[accion][tipo][0]
    with caja.servidor.request() as req:
        yield req
        inicio = env.now
        espera = inicio - llegada
        serv = rng.exponential(media_serv)
        yield env.timeout(serv)
        salida = env.now
        # registrar
        caja.n_clientes += 1
        caja.tiempos_espera.append(espera)
        caja.tiempos_servicio.append(serv)
        caja.tiempos_sistema.append(salida - llegada)
        caja.por_tipo[tipo] += 1
        caja.por_accion[accion] += 1


def generador(env, cajas, rng, media_ll):
    """Genera llegadas al banco y asigna cada cliente a una caja al azar."""
    while True:
        yield env.timeout(rng.exponential(media_ll))
        if env.now > JORNADA:
            break
        # clasificar cliente
        k = rng.choice(len(CATEGORIAS), p=PROBS)
        accion, tipo = CATEGORIAS[k]
        # asignar a una de las 3 cajas (independientes, reparto uniforme)
        caja = cajas[rng.integers(0, N_CAJAS)]
        env.process(cliente(env, caja, accion, tipo, rng))


def corre_una_replica(semilla):
    rng = np.random.default_rng(semilla)
    env = simpy.Environment()
    cajas = [Caja(env, i + 1) for i in range(N_CAJAS)]
    media_ll = media_entre_llegadas_global()
    env.process(generador(env, cajas, rng, media_ll))
    env.run(until=JORNADA + 60)  # deja terminar los que están en servicio
    return cajas, media_ll


def resumen_replica(cajas):
    filas = []
    for c in cajas:
        n = c.n_clientes
        filas.append({
            "Caja": c.idx,
            "Clientes": n,
            "T_espera_prom": np.mean(c.tiempos_espera) if n else 0.0,
            "T_servicio_prom": np.mean(c.tiempos_servicio) if n else 0.0,
            "T_sistema_prom": np.mean(c.tiempos_sistema) if n else 0.0,
            "Rapido": c.por_tipo["Rapido"],
            "Normal": c.por_tipo["Normal"],
            "Lento": c.por_tipo["Lento"],
            "Muy lento": c.por_tipo["Muy lento"],
            "Retiros": c.por_accion["Retiro"],
            "Pagos": c.por_accion["Consignacion o pago"],
        })
    return pd.DataFrame(filas)


if __name__ == "__main__":
    N_REP = 10
    print("Media global entre llegadas (min):",
          round(media_entre_llegadas_global(), 4))

    todas = []
    for r in range(N_REP):
        cajas, _ = corre_una_replica(semilla=1000 + r)
        df = resumen_replica(cajas)
        df.insert(0, "Replica", r + 1)
        todas.append(df)

    detalle = pd.concat(todas, ignore_index=True)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== Detalle por réplica y caja ===")
    print(detalle.round(3).to_string(index=False))

    # Promedios por caja a lo largo de réplicas
    print("\n=== Promedio por caja (todas las réplicas) ===")
    prom_caja = detalle.groupby("Caja").mean(numeric_only=True).drop(columns="Replica")
    print(prom_caja.round(3).to_string())

    detalle.to_csv("detalle_replicas.csv", index=False)
    prom_caja.to_csv("promedio_por_caja.csv")
    print("\nGuardado: detalle_replicas.csv, promedio_por_caja.csv")
