
import numpy as np
import pulp
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import itertools

# Fix seed a reprodukálható eredményekért
np.random.seed(11)

# --- (1) es (2) feladat megoldasa az egyidoszakos modellre vonatkozoan ---

derivativa_ar = []
n_vals = []
min_prices = []
max_prices = []

for fa in range(20,1000, 5):
    # --- 1. Paraméterek és a piac szimulációja ---
    N = fa              # Ágak (scenariók) száma
    S0 = 100            # Kezdőár minden részvényre

    # Lognormális eloszlás paraméterei
    mu = 2
    sigma = 1
    exp_25 = np.exp(2.5)

    # 3 részvény szimulálása 20 ágon
    # (3 sor, 20 oszlop: a sorok a részvények, oszlopok az ágak)
    xi = np.random.lognormal(mean=mu, sigma=sigma, size=(3, N))

    # A t=1 időponti árak kiszámítása (az alpha kiesik)
    S1 = 100 + xi - exp_25

    # Arbitrázs vizsgálat és Kockázatsemleges mérték
    # Olyan q >= 0 valószínűségeket keresünk, ahol a várható jövőbeli ár = kezdőár
    # Készpénz egyenlete biztosítja, hogy a valószínűségek összege 1 legyen.
    A_eq = np.vstack([np.ones(N), S1[0], S1[1], S1[2]])
    b_eq = np.array([1, 100, 100, 100])
    q_bounds = [(0, None) for _ in range(N)]

    # Derivatíva Árazási egyidoszakra LP
    # Opció kifizetése (1-es részvény (index 0) cseréje 2-esre (index 1))
    D = np.maximum(S1[1] - S1[0], 0)

    # PuLP modell inicializálása
    prob = pulp.LpProblem("Derivativa_Arazasa", pulp.LpMinimize)

    # Döntési változók (th_n_j formátumban, n=0 csomópont, j=0,1,2,3 eszköz)
    th_0_0 = pulp.LpVariable("th_0_0") # Készpénz mennyisége
    th_0_1 = pulp.LpVariable("th_0_1") # 1-es részvény
    th_0_2 = pulp.LpVariable("th_0_2") # 2-es részvény
    th_0_3 = pulp.LpVariable("th_0_3") # 3-as részvény

    # Célfüggvény: Kezdeti bekerülési költség minimalizálása (Minden induló ár 100, készpénz ára 1)
    prob += th_0_0 + 100*th_0_1 + 100*th_0_2 + 100*th_0_3, "Kezdeti_Koltseg"

    # Korlátozó feltételek: Szuperreplikáció minden k ágon (t=1 időpontban)
    for k in range(N):
        prob += th_0_0 + th_0_1*S1[0, k] + th_0_2*S1[1, k] + th_0_3*S1[2, k] >= D[k], f"Ag_{k+1}"

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    print("\n--- DERIVATÍVA ÁRAZÁS ---")
    if prob.status == pulp.LpStatusOptimal:
        lp_price = pulp.value(prob.objective)
        print(f"Az LP (szuperreplikációs) módszerrel kapott derivatíva ár: {lp_price:.4f}")
        print("Optimális portfólió összetétele (th_0_j):")
        print(f"  Készpénz (th_0_0): {th_0_0.varValue:.4f}")
        print(f"  1-es részvény (th_0_1): {th_0_1.varValue:.4f}")
        print(f"  2-es részvény (th_0_2): {th_0_2.varValue:.4f}")
        print(f"  3-as részvény (th_0_3): {th_0_3.varValue:.4f}")
        derivativa_ar.append(np.round(lp_price, 4))
    else:
        print("Az LP feladat nem megoldható optimálisan.")

    # Kockázatsemleges mértékkel vett árazás összehasonlítása
    # --- A) MINIMUM ÁR KERESÉSE ---
    # Cél: Minimalizálni q * D-t
    res_min = linprog(c=D, A_eq=A_eq, b_eq=b_eq, bounds=q_bounds, method='highs')

    # --- B) MAXIMUM ÁR KERESÉSE ---
    # Cél: Maximalizálni q * D-t (linprog minimalizál, ezért -D-t adunk meg)
    res_max = linprog(c=-D, A_eq=A_eq, b_eq=b_eq, bounds=q_bounds, method='highs')

    # Ha mindkettő sikeres, akkor nincs arbitrázs, és van érvényes ársáv
    if res_min.success and res_max.success:
        min_p = res_min.fun
        max_p = -res_max.fun  # Visszaalakítjuk a negatív értéket

        n_vals.append(N)
        min_prices.append(min_p)
        max_prices.append(max_p)

        print(f"N = {N:4d} | Arbitrázsmentes sáv: [ {min_p:7.4f} , {max_p:7.4f} ]")
    else:
        print(f"N = {N:4d} | Arbitrázs van a piacon (nincs RNM).")

# --- Grafikon kirajzolása ---
plt.figure(figsize=(10, 6))

# Kitöltött sáv rajzolása a minimum és maximum között
plt.fill_between(n_vals, min_prices, max_prices, color='skyblue', alpha=0.4, label='Arbitrázsmentes ársáv')

# Határvonalak
plt.plot(n_vals, max_prices, color='blue', linestyle='--', linewidth=1, label='Max ár')
plt.plot(n_vals, min_prices, color='red', linestyle='--', linewidth=1, label='Min ár')

plt.title('Kockázatsemleges mértékkel vett árazás az ágak számának (N) függvényében')
plt.xlabel('Ágak száma (N)')
plt.ylabel('Kockázatsemleges mértékkel vett ár')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()

#Ársáv plot
plt.savefig("arsav.png")
plt.show()

#Derivatíva ár plot
n_ek = range(20, 1000, 5)
plt.plot(n_ek, derivativa_ar)
plt.savefig("derivativa_ar.png")
plt.show()

#LP kiírása
#prob.writeLP("Opkut.lp")

# (3) es (4) feladat megoldasa a ketidoszakos modellre vonatkozoan

def check_arbitrage_node(S_current, S_children):

    #Megvizsgálja, van-e arbitrázs a th_n_j döntési változókkal.
    #Visszatérési érték: True, ha VAN arbitrázs, False ha NINCS (arbitrázsmentes).

    num_assets = len(S_current)
    num_children = len(S_children)

    # Döntési változók (th_n_0, th_n_1, th_n_2, th_n_3)
    c = np.zeros(num_assets)

    A_ub = []
    b_ub = []

    # 1. Bekerülési költség <= 0
    A_ub.append(S_current)
    b_ub.append(0)

    # 2. Minden gyermek csomópontban a kifizetés >= 0 (azaz -kifizetés <= 0)
    for c_idx in range(num_children):
        A_ub.append(-S_children[c_idx])
        b_ub.append(0)

    # 3. Szigorúan pozitív haszon legalább egy állapotban
    # Ennek eléréséhez a kifizetések összegét >= 1-re kényszerítjük
    A_ub.append(-np.sum(S_children, axis=0))
    b_ub.append(-1)

    # Korlátok a th_n_j változókra: nincsenek (lehet shortolni is)
    bounds = [(None, None) for _ in range(num_assets)]

    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method='highs')

    return res.success


def generate_and_check_tree(alpha, n):
    S = {}
    children = {}

    # t=0
    S[0] = np.array([1.0, 100.0, 100.0, 100.0])
    children[0] = list(range(1, n + 1))

    # t=1
    node_idx = 1
    for _ in range(n):
        xi = np.random.lognormal(2.0, 1.0, 3)
        S1_stocks = alpha * 100.0 + (1 - alpha) * S[0][1:] + xi - np.exp(2.5)
        S[node_idx] = np.array([1.0, S1_stocks[0], S1_stocks[1], S1_stocks[2]])
        # A 2. időszak csomópontjai n+1-től kezdődnek
        children[node_idx] = list(range(n + 1 + (node_idx - 1) * n, n + 1 + node_idx * n))
        node_idx += 1

    # t=2
    for parent in range(1, n + 1):
        for child in children[parent]:
            xi = np.random.lognormal(2.0, 1.0, 3)
            S2_stocks = alpha * 100.0 + (1 - alpha) * S[parent][1:] + xi - np.exp(2.5)
            S[child] = np.array([1.0, S2_stocks[0], S2_stocks[1], S2_stocks[2]])
            children[child] = []

    # Arbitrázs vizsgálat a t=0 és t=1 csomópontokon
    if check_arbitrage_node(S[0], np.array([S[c] for c in children[0]])):
        return False  # Nem arbitrázsmentes

    for i in range(1, n + 1):
        if check_arbitrage_node(S[i], np.array([S[c] for c in children[i]])):
            return False  # Nem arbitrázsmentes

    return True  # Arbitrázsmentes


# --- SZIMULÁCIÓ FUTTATÁSA ---
alphas = [0.0, 0.3, 0.5, 0.7, 0.9]
ns = [5, 15, 20, 50, 100, 200]
num_trials = 100

print("Arbitrázsmentes folyamatok aránya:")
for alpha, n in itertools.product(alphas, ns):
    arbitrage_free_count = sum([generate_and_check_tree(alpha, n) for _ in range(num_trials)])
    print(f"Alpha: {alpha:3.1f} | n: {n:2d} -> Arány: {arbitrage_free_count / num_trials:.0%}")


def price_two_period_derivative(alpha, n):
    # Rögzített seed a stabil magyarázathoz


    # FA GENERÁLÁSA ÉS STRUKTÚRA FELÉPÍTÉSE
    S = {}
    children = {}  # Itt tároljuk, melyik csomópontnak kik a gyerekei

    # t=0 (Gyökér)
    S[0] = np.array([1.0, 100.0, 100.0, 100.0])
    children[0] = list(range(1, n + 1))  # A 0. csomópont gyerekei az 1..n indexek

    # t=1 csomópontok generálása
    for k in range(1, n + 1):
        xi = np.random.lognormal(2.0, 1.0, 3)
        # alpha itt kiesik, mert S0=100
        S_stocks = 100.0 + xi - np.exp(2.5)
        S[k] = np.array([1.0, S_stocks[0], S_stocks[1], S_stocks[2]])
        children[k] = []  # Inicializáljuk a t=1 csomópont gyerekeinek listáját

    # t=2 csomópontok (levelek) generálása
    leaf_idx = n + 1
    parents = {}
    for k in range(1, n + 1):
        for _ in range(n):
            xi = np.random.lognormal(2.0, 1.0, 3)
            S_stocks = alpha * 100.0 + (1 - alpha) * S[k][1:] + xi - np.exp(2.5)
            S[leaf_idx] = np.array([1.0, S_stocks[0], S_stocks[1], S_stocks[2]])

            parents[leaf_idx] = k  # Szuperreplikációhoz kell
            children[k].append(leaf_idx)  # Arbitrázs-ellenőrzéshez kell
            leaf_idx += 1

    # SZIGORÚ ARBITRÁZS-ELLENŐRZÉS
    is_free = True

    # t=0 szint ellenőrzése
    if check_arbitrage_node(S[0], np.array([S[c] for c in children[0]])):
        is_free = False

    # t=1 szint ellenőrzése (minden k csomópontra)
    if is_free:
        for k in range(1, n + 1):
            if check_arbitrage_node(S[k], np.array([S[c] for c in children[k]])):
                is_free = False
                break

    # ÁRAZÁS CSAK HA A PIAC TISZTA (ARBITRÁZSMENTES)
    if not is_free:
        return None  # Ha van arbitrázs, nem definiálunk árat

    # PuLP modell inicializálása
    prob = pulp.LpProblem("2_Period_SuperReplication", pulp.LpMinimize)

    # Döntési változók: th_node_asset
    th = {}
    for node in range(n + 1):  # 0. és 1..n csomópontok
        for j in range(4):  # 4 eszköz (0:cash, 1,2,3:részvények)
            th[(node, j)] = pulp.LpVariable(f"th_{node}_{j}")

    # Célfüggvény: Kezdeti költség t=0-ban
    prob += pulp.lpSum([th[(0, j)] * S[0][j] for j in range(4)])

    # Feltétel 1: Önfinanszírozás t=1-ben
    for k in range(1, n + 1):
        val_old = pulp.lpSum([th[(0, j)] * S[k][j] for j in range(4)])
        cost_new = pulp.lpSum([th[(k, j)] * S[k][j] for j in range(4)])
        prob += val_old >= cost_new

    # Feltétel 2: Szuperreplikáció t=2-ben
    for m in range(n + 1, leaf_idx):
        p = parents[m]
        val_terminal = pulp.lpSum([th[(p, j)] * S[m][j] for j in range(4)])
        # Derivatíva kifizetése (pl. 2-es részvény mínusz 1-es részvény)
        D_m = max(S[m][2] - S[m][1], 0)
        prob += val_terminal >= D_m

    # Megoldás
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    #prob.writeLP("ketto_idoszak_deviza.lp")
    if prob.status == pulp.LpStatusOptimal:
        return pulp.value(prob.objective)
    return None

# --- SZIMULÁCIÓ ÉS ELEMZÉS ---
alphas = [0.0, 0.3, 0.5, 0.7, 0.9]
ns = [5, 15, 20, 50, 100, 200]

print("Derivatíva árazása (Szuperreplikáció) a paraméterek függvényében:\n")
for n in ns:
    for alpha in alphas:
        price = price_two_period_derivative(alpha, n)
        if price is not None:
            # Ha az ár masszívan negatív, az arbitrázst jelent
            if price < -1e5:
                print(f"n: {n:2d} | alpha: {alpha:3.1f} -> Ár: ARBITRÁZS")
            else:
                print(f"n: {n:2d} | alpha: {alpha:3.1f} -> Ár: {price:8.4f}")
        else:
            print(f"n: {n:2d} | alpha: {alpha:3.1f} -> Ár: ARBITRÁZS / NEM MEGOLDHATÓ")
