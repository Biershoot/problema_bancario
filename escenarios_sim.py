"""
Simula los escenarios de especialización para confirmar el análisis teórico:
  A) 2 cajas exclusivas retiros + 1 caja exclusiva pagos
  B) 1 caja exclusiva retiros + 2 cajas exclusivas pagos
  C) Actual: 3 cajas mixtas (reparto uniforme)
Cada caja es M/M/1 independiente.
"""
import numpy as np
import pandas as pd
import simpy
from simulacion import (TABLA1, TABLA2, P_RETIRO, P_PAGO, JORNADA, TIPOS,
                        media_entre_llegadas_global)

MEDIA_LL = media_entre_llegadas_global()
TIPOS_RET = list(TABLA2["Retiro"].items())
TIPOS_PAG = list(TABLA2["Consignacion o pago"].items())
P_RET = np.array([p for _, p in TIPOS_RET]); P_RET /= P_RET.sum()
P_PAG = np.array([p for _, p in TIPOS_PAG]); P_PAG /= P_PAG.sum()


class Caja:
    def __init__(self, env):
        self.s = simpy.Resource(env, capacity=1)
        self.esp = []; self.sis = []; self.n = 0


def cliente(env, caja, media_serv, rng):
    t0 = env.now
    with caja.s.request() as req:
        yield req
        esp = env.now - t0
        yield env.timeout(rng.exponential(media_serv))
        caja.n += 1; caja.esp.append(esp); caja.sis.append(env.now - t0)


def gen(env, cajas_ret, cajas_pag, rng):
    while True:
        yield env.timeout(rng.exponential(MEDIA_LL))
        if env.now > JORNADA:
            break
        if rng.random() < P_RETIRO:  # retiro
            k = rng.choice(len(TIPOS_RET), p=P_RET)
            tipo = TIPOS_RET[k][0]
            ms = TABLA1["Retiro"][tipo][0]
            caja = cajas_ret[rng.integers(0, len(cajas_ret))]
        else:
            k = rng.choice(len(TIPOS_PAG), p=P_PAG)
            tipo = TIPOS_PAG[k][0]
            ms = TABLA1["Consignacion o pago"][tipo][0]
            caja = cajas_pag[rng.integers(0, len(cajas_pag))]
        env.process(cliente(env, caja, ms, rng))


def corre(nret, npag, semilla):
    rng = np.random.default_rng(semilla)
    env = simpy.Environment()
    cr = [Caja(env) for _ in range(nret)]
    cp = [Caja(env) for _ in range(npag)]
    env.process(gen(env, cr, cp, rng))
    env.run(until=JORNADA + 120)
    def agg(cajas):
        esp = np.concatenate([c.esp for c in cajas]) if any(c.esp for c in cajas) else np.array([0.0])
        sis = np.concatenate([c.sis for c in cajas]) if any(c.sis for c in cajas) else np.array([0.0])
        n = sum(c.n for c in cajas)
        return n, esp.mean(), sis.mean()
    return agg(cr), agg(cp)


def escenario(nret, npag, nrep=10):
    rows = []
    for r in range(nrep):
        (nr, wqr, wr), (np_, wqp, wp) = corre(nret, npag, 2000 + r)
        rows.append([nr, wqr, wr, np_, wqp, wp])
    a = np.array(rows)
    return {
        "nret": nret, "npag": npag,
        "Wq_ret": a[:,1].mean(), "W_ret": a[:,2].mean(),
        "Wq_pag": a[:,4].mean(), "W_pag": a[:,5].mean(),
        "n_ret": a[:,0].mean(), "n_pag": a[:,3].mean(),
    }


print("Simulación de escenarios de especialización (10 réplicas c/u):\n")
for nret, npag in [(2,1),(1,2)]:
    r = escenario(nret, npag)
    print(f"--- {nret} caja(s) RETIRO + {npag} caja(s) PAGO ---")
    print(f"  Retiros: Wq={r['Wq_ret']:.2f} min, W={r['W_ret']:.2f} min, n={r['n_ret']:.0f}")
    print(f"  Pagos  : Wq={r['Wq_pag']:.2f} min, W={r['W_pag']:.2f} min, n={r['n_pag']:.0f}")
    print()
