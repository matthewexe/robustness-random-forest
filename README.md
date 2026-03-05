# Robustness of Random Forests on Multiclass Data

## Panoramica del progetto

Questo progetto implementa un sistema per il calcolo della **robustezza** di un classificatore *Random Forest* su dati multiclasse.

Il progetto si sviluppa in due componenti principali:

1. **`Random_Forest_Aeon_Univariate/`** — componente per generare random forest su dataset aeon. 
2. **`src/robustness/`** — legge i file prodotti dal componente precedente e calcola la robustezza applicando il *Tableau Method* su una formula Booleana che codifica la foresta.

---

### Endpoint Universe

L'*Endpoint Universe* $\mathcal{EU}$ è un dizionario che associa ad ogni feature $f$ una soglia $\theta_f \in \mathbb{R}$. Questa soglia funge da punto di bisezione dello spazio continuo della feature: ogni valore $x_f \leq \theta_f$ è codificato come il bit `0`, ogni valore $x_f > \theta_f$ è codificato come il bit `1`.

### Encoding Booleano della Random Forest (PSF)

Per poter ragionare formalmente sulla foresta, la Random Forest viene convertita in una formula booleana chiamata **PSF** (*Propositional Set Formula*). La conversione avviene enumerando tutti i cammini radice-foglia di tutti gli alberi.

Ogni cammino produce una congiunzione di letterali booleani. La formula complessiva è una disgiunzione di tali congiunzioni, nella forma:

```
PSF = (l₁₁ ∧ l₁₂ ∧ … ∧ cK₁) ∨ (l₂₁ ∧ l₂₂ ∧ … ∧ cK₂) ∨ …
```

dove ogni `lᵢⱼ` è un letterale su una variabile di feature (o la sua negazione) e `cKᵢ` è la variabile di **classe** associata alla foglia (ad esempio `c1` per la classe `1`).


### Ordered Binary Decision Diagram (OBDD)

Un **OBDD** (*Ordered Binary Decision Diagram*) è una rappresentazione compatta e canonica di una funzione booleana. I nodi interni sono etichettati con variabili booleane e hanno due archi figli: *low* (variabile = 0) e *high* (variabile = 1). I nodi terminali sono `True` (⊤) e `False` (⊥).
Nel progetto è stata utilizzata la libreria `dd` per gestire gli OBDD.

Esempio di OBDD per la formula `(x1 ∧ ¬x2) ∨ (¬x1 ∧ x2)` (XOR):

```mermaid
graph TD
    x1((x1))
    x2a((x2))
    x2b((x2))
    T(("⊤"))
    F(("⊥"))

    x1 -->|"0 (low)"| x2a
    x1 -->|"1 (high)"| x2b
    x2a -->|"0"| F
    x2a -->|"1"| T
    x2b -->|"0"| T
    x2b -->|"1"| F
```


### Riduzione parziale della PSF (`partial_reduce`)

Poiché la PSF può essere molto grande, non è sempre possibile ridurla direttamente a un singolo OBDD. La funzione `partial_reduce` applica un approccio **bottom-up**: visita l'albero della formula in post-ordine e converte i sottoalberi in OBDD solo se la dimensione risultante è al di sotto del parametro `diagram_size`. Se la dimensione supera la soglia, il nodo mantiene la sua forma.

`partial_reduce` può anche ricevere un'assegnazione parziale di variabili: in tal caso specializza la formula applicando l'assegnazione prima della riduzione.

```python
# src/robustness/domain/psf/operations.py  (estratto semplificato)

def partial_reduce(psf: PSF, diagram_size: int, assignment: dict[str, bool] = None):
    for psf_node_id in psf.postorder_iter():     # visita bottom-up
        kind = psf.get_kind_of(psf_node_id)

        if kind == Kind.AND:
            # Tenta di unire i due figli in un OBDD tramite AND booleano
            if left_is_bdd and right_is_bdd:
                new_bdd = manager.apply("and", bdd_left, bdd_right)
                # Tieni il risultato solo se non supera la soglia
                outcome = new_bdd.dag_size <= diagram_size
                ...
            else:
                # Almeno un figlio non è ancora un OBDD: rimane come nodo AST
                node_id = builder.And(left_id, right_id)
        ...
    return reduced_tree, last_outcome
```

### Tableau Method

Il **Tableau Method** estende la riduzione parziale con una strategia di *splitting*: si costruisce un albero (il *Tableau Tree*) dove ogni nodo contiene una PSF parzialmente ridotta.

L'algoritmo opera come segue:

1. Si applica `partial_reduce` alla PSF iniziale.
2. Se il risultato è già un singolo OBDD, il nodo diventa una **foglia** del tableau.
3. Altrimenti, si sceglie la variabile di feature più frequente negli OBDD della formula (*best feature*) e si creano due nodi figli:
   - **ramo low**: la variabile è assegnata a `False`
   - **ramo high**: la variabile è assegnata a `True`
4. Per ciascun figlio si riesegue `partial_reduce` con l'assegnazione della variabile prescelta.
5. Il processo continua ricorsivamente fino a quando ogni foglia contiene un singolo OBDD.

Esempio di Tableau Tree dopo lo splitting su `x3`, poi su `x1`:

```mermaid
graph TD
    R["PSF ridotta (nodo 0)"]
    L["PSF|x3=False (nodo 1)<br>OBDD"]
    H["PSF|x3=True (nodo 2)"]
    HL["PSF|x3=True,x1=False<br>OBDD (nodo 3)"]
    HH["PSF|x3=True,x1=True<br>OBDD (nodo 4)"]

    R -->|"x3 = False, c=0/1"| L
    R -->|"x3 = True,  c=0/1"| H
    H -->|"x1 = False, c=0/1"| HL
    H -->|"x1 = True,  c=0/1"| HH
```

Implementazione del loop principale del tableau:

```python
# src/robustness/domain/psf/operations.py

def tableau_method(f: PSF) -> TableauTree:
   tree = tb.Builder()
   root = tree.add_psf(f, "Initial Reduced-PSF")
   frontier = deque([root])

   while frontier:
      current = frontier.pop()
      current_psf = tree.current_tree.get_psf_of(current)

      if is_bdd(current_psf):
         continue  # foglia: già un OBDD

      best_var, _ = best_feature(current_psf)  # variabile più frequente

      # Ramo low: assegna best_var = False
      low_tree, _ = partial_reduce(current_psf, config.diagram_size, {best_var: False})
      low_id = tree.add_psf(low_tree, best_feature(low_tree)[0])
      tree.assign(current, low_id, best_var, False)
      frontier.append(low_id)

      # Ramo high: assegna best_var = True
      high_tree, _ = partial_reduce(current_psf, config.diagram_size, {best_var: True})
      high_id = tree.add_psf(high_tree, best_feature(high_tree)[0])
      tree.assign(current, high_id, best_var, True)
      frontier.append(high_id)

   return tree.build()
```

### Calcolo della robustezza su un OBDD

Dato un sample $s$ e un OBDD $f$, la robustezza sull'OBDD è calcolata come segue:

1. Si costruisce un **DAG di robustezza** a partire dall'OBDD assegnando i pesi agli archi tramite confronto con l'*Endpoint Universe* (EU):
   - Nodo feature $x_i$: se $s_{x_i} \leq \mathcal{EU}(x_i)$ → ramo **low** peso `0`, ramo **high** peso `1`; se $s_{x_i} > \mathcal{EU}(x_i)$ → ramo **high** peso `0`, ramo **low** peso `1`.
   - Nodo classe: ramo **high** → `⊤` (peso `1`), ramo **low** → `⊥` (peso `+∞`).
   - Il ramo *high* del nodo corrispondente alla **classe predetta** da $s$ viene rimosso.
2. La robustezza è la **lunghezza del cammino minimo** nel DAG pesato dal nodo radice al nodo `⊤`.

Esempio con sample $s = \{x_1{:}5.0,\; x_2{:}1.0,\; x_3{:}8.0,\; x_4{:}0.5,\; x_5{:}9.0\}$,
soglie EU $= \{x_1{:}3.5,\; x_2{:}2.5,\; x_3{:}5.0,\; x_4{:}2.0,\; x_5{:}7.0\}$, classe predetta `c2`.

**Passo 1 — OBDD originale** (percorso del sample evidenziato in blu):

```mermaid
graph TD
    x1(("x1"))
    x2(("x2"))
    x3(("x3"))
    x4(("x4"))
    x5(("x5"))
    c1(("c1"))
    c2(("c2"))
    c3(("c3"))
    T(("⊤"))
    F(("⊥"))

    x1 -->|"low"| x5
    x1 ==>|"high ✓"| x2

    x5 -->|"low"| F
    x5 -->|"high"| c1

    x2 ==>|"low ✓"| x3
    x2 -->|"high"| x5

    x3 -->|"low"| c1
    x3 ==>|"high ✓"| x4

    x4 ==>|"low ✓"| c2
    x4 -->|"high"| c3

    c1 -->|"low"| F
    c1 -->|"high"| T

    c2 -->|"low"| F
    c2 ==>|"high ✓"| T

    c3 -->|"low"| F
    c3 -->|"high"| T

    style F fill:#f88,stroke:#c00
    style T fill:#8f8,stroke:#080
    style x1 stroke:#00c,stroke-width:3px
    style x2 stroke:#00c,stroke-width:3px
    style x3 stroke:#00c,stroke-width:3px
    style x4 stroke:#00c,stroke-width:3px
    style c2 stroke:#00c,stroke-width:3px
```

> **Percorso del sample (archi doppi, bordo blu):** `x1 --high→ x2 --low→ x3 --high→ x4 --low→ c2 --high→ ⊤`

---

**Passo 2 — pesatura degli archi via EU** → costruzione del DAG di robustezza

Confronti EU che determinano i pesi (arco con confronto vero → w=0):

| nodo | confronto        | low | high |
|------|------------------|-----|------|
| x1   | 5.0 > EU=3.5     | w=1 | **w=0** |
| x2   | 1.0 ≤ EU=2.5     | **w=0** | w=1 |
| x3   | 8.0 > EU=5.0     | w=1 | **w=0** |
| x4   | 0.5 ≤ EU=2.0     | **w=0** | w=1 |
| x5   | 9.0 > EU=7.0     | w=1 | **w=0** |

```mermaid
graph TD
    x1["x1  EU=3.5"]
    x2["x2  EU=2.5"]
    x3["x3  EU=5.0"]
    x4["x4  EU=2.0"]
    x5["x5  EU=7.0"]
    c1(("c1"))
    c2(("c2"))
    c3(("c3"))
    T(("⊤"))
    F(("⊥ ∞"))

    x1 -->|"low  w=1"| x5
    x1 -->|"high w=0"| x2

    x5 -->|"low  w=1"| F
    x5 -->|"high w=0"| c1

    x2 -->|"low  w=0"| x3
    x2 -->|"high w=1"| x5

    x3 -->|"low  w=1"| c1
    x3 -->|"high w=0"| x4

    x4 -->|"low  w=0"| c2
    x4 -->|"high w=1"| c3

    c1 -->|"low  w=∞"| F
    c1 -->|"high w=1"| T

    c2 -->|"low  w=∞"| F
    %% ramo high di c2 rimosso: è la classe predetta

    c3 -->|"low  w=∞"| F
    c3 -->|"high w=1"| T

    style F fill:#f88,stroke:#c00
    style T fill:#8f8,stroke:#080
    style x1 stroke:#00c,stroke-width:3px
    style x2 stroke:#00c,stroke-width:3px
    style x3 stroke:#00c,stroke-width:3px
    style x4 stroke:#00c,stroke-width:3px
    style c2 stroke:#c00,stroke-width:3px,stroke-dasharray:5 5
```

> **Percorso del sample (nodi con bordo blu):** `x1 --high(w=0)→ x2 --low(w=0)→ x3 --high(w=0)→ x4 --low(w=0)→ c2` → bloccato (ramo high di `c2` rimosso).
> Il cammino minimo verso `⊤` è p.es. `x1→x2→x3→x4--high(w=1)→c3--high(w=1)→⊤` con costo `0+0+0+1+1 = 2`.
> **Robustezza = 2**: occorre perturbare il valore di almeno 2 feature affinché il sample venga classificato diversamente da `c2`.

Tracciamento del percorso del sample nell'OBDD:

```python
# src/robustness/domain/bdd/operations.py

def test_sample(sample: Sample, manager: DD_Manager, f: DD_Function, endpoints: Endpoints) -> str:
    """Restituisce la stringa binaria del percorso radice→True nell'OBDD."""
    if f in {manager.false, manager.true}:
        return ""
    if is_class(f.var):
        # Variabile di classe: segue il ramo high se la classe predetta corrisponde
        if sample.predicted_label == f.var[1:]:
            return test_sample(sample, manager, f.high, endpoints) + "1"
        else:
            return test_sample(sample, manager, f.low, endpoints) + "0"
    # Variabile di feature: confronta con la soglia dell'Endpoint Universe
    if sample.features[f.var] <= endpoints[f.var]:
        return test_sample(sample, manager, f.low, endpoints) + "0"
    else:
        return test_sample(sample, manager, f.high, endpoints) + "1"
```

Calcolo della robustezza sull'OBDD tramite shortest path:

```python
# src/robustness/domain/mappers/bdd.py

def calculate_bdd_robustness(f: DD_Function, sample: Sample, endpoints: Endpoints) -> float:
    manager = get_bdd_manager()
    if f == manager.true:   return 0        # già True: robustezza 0
    if f == manager.false:  return math.inf # mai True: robustezza infinita

    path = test_sample(sample, manager, f, endpoints)
    dag  = construct_robustness_dag(manager, f, sample, path)

    # Dijkstra / BFS sul DAG pesato
    shortest_path = nx.shortest_path(dag, dag.root, dag.true(), weight="weight")
    return sum(dag[u][v].get('weight', 1) for u, v in zip(shortest_path, shortest_path[1:]))
```

### Calcolo della robustezza sul Tableau

La robustezza sul Tableau Tree è la robustezza minima calcolata sull'intero albero. Per ciascuna foglia (che contiene un OBDD), si calcola la robustezza sull'OBDD e si propaga il costo verso la radice sommando i costi degli archi del tableau.

Il costo di un arco nel tableau è definito dalla funzione $c(s, \mathcal{EU}, \text{var}, \text{asgn})$:

$$
c(s, \mathcal{EU}, \text{var}, \text{asgn}) = \begin{cases} 0 & \text{se asgn} = \text{False} \land s[\text{var}] \leq \mathcal{EU}[\text{var}] \\ 0 & \text{se asgn} = \text{True} \land s[\text{var}] > \mathcal{EU}[\text{var}] \\ 1 & \text{altrimenti} \end{cases}
$$

In parole: il costo è `0` se l'assegnazione del ramo tableau è **coerente** con il valore reale del campione (ovvero il campione si troverebbe naturalmente su quel ramo), e `1` altrimenti (il campione deve essere perturbato per seguire quel ramo).

La robustezza finale è:

$$
\rho(s) = \min_{\ell \in \text{foglie}} \left( \rho_{\text{OBDD}}(\ell, s) + \sum_{(u,v) \in \pi(r, \ell)} c(s, \mathcal{EU}, \text{var}_{(u,v)}, \text{asgn}_{(u,v)}) \right)
$$

dove $\pi(r, \ell)$ è il percorso dalla radice alla foglia $\ell$ nel Tableau Tree.

Implementazione della funzione di costo e propagazione:

```python
# src/robustness/domain/psf/operations.py

def robustness(t: TableauTree, sample: Sample, endpoints: Endpoints) -> int:
    memo = {}

    for leaf in t.leaves:
        leaf_psf = t.get_psf_of(leaf)
        bdd = leaf_psf.get_value_of(leaf_psf.root_id)
        memo[leaf] = calculate_bdd_robustness(bdd, sample, endpoints)

        parent = t.parent(leaf)
        current = leaf
        while parent is not None:
            var, assignment = t.get_edge_assignment(parent, current)
            path_cost  = memo[current]

            if not assignment:
                # ramo low: c = 0 se sample[var] <= EU[var], altrimenti 1
                path_cost += 0 if sample.features[var] <= endpoints[var] else 1
            else:
                # ramo high: c = 0 se sample[var] > EU[var], altrimenti 1
                path_cost += 0 if sample.features[var] > endpoints[var] else 1

            # Prendi il minimo tra i costi già visti per questo nodo
            memo[parent] = min(memo.get(parent, float('inf')), path_cost)
            current = parent
            parent = t.parent(current)

    return memo[t.root_id]
```

---

## Architettura del sistema e pipeline

```mermaid
flowchart TD
    A["📂 Random Forest JSON<br>Endpoint Universe JSON<br>Sample JSON"]
    B["RandomForestService<br><i>adapters/rf_service.py</i>"]
    C["rf_to_formula_str<br><i>mappers/rf.py</i>"]
    D["parse_psf<br><i>psf/parser.py</i>"]
    E["partial_reduce<br><i>psf/operations.py</i>"]
    F["tableau_method<br><i>psf/operations.py</i>"]
    G["robustness<br><i>psf/operations.py</i>"]
    H["🔢 Robustezza: intero<br>(numero di bit da modificare)"]

    A --> B
    B -->|"RandomForest + Sample + Endpoints"| C
    C -->|"stringa PSF"| D
    D -->|"AST BinaryTree"| E
    E -->|"PSF ridotta (OBDD + AST)"| F
    F -->|"TableauTree"| G
    G --> H
```

### Descrizione dei moduli principali

| Modulo | Percorso | Responsabilità |
|---|---|---|
| `RandomForestService` | `src/robustness/adapters/rf_service.py` | Lettura dei file JSON prodotti dal componente del professore |
| `rf_to_formula_str` | `src/robustness/domain/mappers/rf.py` | Conversione RF → formula PSF |
| `parse_psf` | `src/robustness/domain/psf/parser.py` | Parsing della formula PSF → AST |
| `partial_reduce` | `src/robustness/domain/psf/operations.py` | Riduzione parziale PSF → OBDD |
| `tableau_method` | `src/robustness/domain/psf/operations.py` | Costruzione del Tableau Tree |
| `robustness` | `src/robustness/domain/psf/operations.py` | Calcolo della robustezza sul Tableau |
| `calculate_bdd_robustness` | `src/robustness/domain/mappers/bdd.py` | Calcolo della robustezza su un singolo OBDD |

---

## Struttura del repository

```
robustness-random-forest/
├── Random_Forest_Aeon_Univariate/   # Componente del professore
│   ├── init_aeon_univariate.py      # Script per addestrare la RF e generare i file
│   ├── forest.py                    # Struttura dati Random Forest
│   ├── tree.py                      # Struttura dati Decision Tree
│   ├── eu.py                        # Calcolo dell'Endpoint Universe
│   ├── sample.py                    # Serializzazione dei campioni
│   └── results/                     # Output: RF, endpoint universe, sample JSON
│
├── src/
│   └── robustness/
│       ├── __main__.py              # Entry point CLI
│       ├── adapters/                # Lettura file (RandomForestService)
│       ├── domain/
│       │   ├── random_forest.py     # Modelli di dominio (RandomForest, Sample, Endpoints)
│       │   ├── bdd/                 # Gestione OBDD (dd library wrapper)
│       │   ├── psf/
│       │   │   ├── model.py         # Modello PSF (AST con Builder)
│       │   │   ├── parser.py        # Parser formula PSF
│       │   │   ├── operations.py    # partial_reduce, tableau_method, robustness
│       │   │   └── tableau/         # Modello TableauTree
│       │   ├── mappers/
│       │   │   ├── rf.py            # RF → formula PSF
│       │   │   ├── psf.py           # stringa → PSF
│       │   │   └── bdd.py           # Calcolo robustezza su OBDD
│       │   └── tree/                # Struttura BinaryTree generica
│       └── schemas/                 # Schemi Pydantic per i file JSON
│
├── tests/                           # Test unitari
├── pyproject.toml                   # Configurazione progetto e dipendenze
└── README.md
```

---

## Installazione

### Con Poetry (consigliato)

```bash
# Installare Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Installare le dipendenze
poetry install
```

### Con venv

```bash
# Creare un ambiente virtuale
python3 -m venv venv

# Attivare l'ambiente
# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\activate

# Installare le dipendenze
pip install -r requirements.txt
```

---

## Utilizzo

### 1. Generazione della Random Forest (componente del professore)

```bash
cd Random_Forest_Aeon_Univariate

# Elencare i dataset disponibili
python init_aeon_univariate.py --list-datasets

# Addestrare la RF sul dataset "Coffee" (senza ottimizzazione)
python init_aeon_univariate.py Coffee

# Addestrare la RF con ottimizzazione Bayesiana degli iperparametri
python init_aeon_univariate.py Coffee --optimize
```

I risultati vengono salvati nella cartella `results/` con i seguenti file:
- `Coffee_random_forest.json` — la Random Forest serializzata
- `Coffee_endpoints_universe.json` — l'Endpoint Universe
- `sample_meta_Coffee_<group>_<id>.json` — i campioni di test

### 2. Calcolo della robustezza

```bash
# Con Poetry
poetry run robustness \
  --dataset-name Coffee \
  --rf-path ./Random_Forest_Aeon_Univariate/results \
  --sample-group 1 \
  --sample-id 0 \
  --diagram-size 50

# Oppure direttamente
python -m robustness \
  --dataset-name Coffee \
  --rf-path ./Random_Forest_Aeon_Univariate/results \
  --sample-group 1 \
  --sample-id 0
```

### Parametri CLI principali

| Parametro | Descrizione | Default |
|---|---|---|
| `--dataset-name`, `-dn` | Nome del dataset Aeon usato per addestrare la RF | `Meat` |
| `--rf-path` | Cartella contenente i file JSON della RF | `./Random_Forest_Aeon_Univariate/results` |
| `--sample-group` | Gruppo del campione da testare | `1` |
| `--sample-id` | ID del campione nel gruppo | `0` |
| `--diagram-size`, `-dd` | Dimensione massima dell'OBDD nella riduzione parziale | `50` |
| `--log-graphs` | Salva i grafi SVG nella cartella `logs/` | `False` |
| `--debug` | Abilita la modalità debug (include `--log-graphs`) | `False` |
