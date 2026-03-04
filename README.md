# Robustness of Random Forests on Multiclass Data

## Panoramica del progetto

Questo progetto implementa un sistema per il calcolo della **robustezza** di un classificatore *Random Forest* su dati multiclasse. L'obiettivo è determinare, dato un campione classificato correttamente dalla foresta, la distanza minima (in termini di perturbazioni binarie sulle feature) che è necessario percorrere nello spazio delle feature per ottenere una classificazione diversa.

Il progetto si sviluppa in due componenti principali:

1. **`Random_Forest_Aeon_Univariate/`** — componente del professore: addestra una Random Forest su dataset di serie temporali univariate (tramite la libreria [Aeon](https://www.aeon-toolkit.org/)) e serializza la foresta, i campioni di test e l'*endpoint universe* in formato JSON.
2. **`src/robustness/`** — componente sviluppata in questo lavoro: legge i file prodotti dal componente precedente e calcola la robustezza applicando il *Tableau Method* su una formula Booleana che codifica la foresta.

---

## Background teorico

### Random Forest

Una *Random Forest* è un insieme di $n$ alberi decisionali $\{T_1, \ldots, T_n\}$, ognuno dei quali partiziona lo spazio delle feature tramite soglie sulle variabili di input. La predizione della foresta è determinata dal voto di maggioranza fra le predizioni dei singoli alberi.

Ogni cammino radice-foglia in un albero $T_i$ definisce una congiunzione di condizioni booleane della forma $x_j \leq \theta$ (ramo *low*) oppure $x_j > \theta$ (ramo *high*).

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

La dimensione di un OBDD è misurata dal numero di nodi nel DAG (`dag_size`). La gestione degli OBDD è affidata alla libreria [`dd`](https://github.com/tulip-control/dd).

### Riduzione parziale della PSF (`partial_reduce`)

Poiché la PSF può essere molto grande, non è sempre possibile ridurla direttamente a un singolo OBDD. La funzione `partial_reduce` applica un approccio **bottom-up**: visita l'AST della formula in post-ordine e converte i sottoalberi in OBDD solo se la dimensione risultante è al di sotto del parametro `diagram_size`. Se la dimensione supera la soglia, il nodo rimane nella forma simbolica (AST).

`partial_reduce` può anche ricevere un'assegnazione parziale di variabili: in tal caso specializza la formula applicando l'assegnazione prima della riduzione.

### Tableau Method

Il **Tableau Method** estende la riduzione parziale con una strategia di *splitting*: si costruisce un albero (il *Tableau Tree*) dove ogni nodo contiene una PSF parzialmente ridotta.

L'algoritmo opera come segue:

1. Si applica `partial_reduce` alla PSF iniziale.
2. Se il risultato è già un singolo OBDD, il nodo è una foglia del tableau.
3. Altrimenti, si sceglie la variabile di feature più frequente negli OBDD della formula (*best feature*) e si creano due nodi figli:
   - **ramo low**: la variabile è assegnata a `False`
   - **ramo high**: la variabile è assegnata a `True`
4. Per ciascun figlio si riesegue `partial_reduce` con l'assegnazione della variabile prescelta.
5. Il processo continua ricorsivamente fino a quando ogni foglia contiene un singolo OBDD.

### Calcolo della robustezza su un OBDD

Dato un sample $s$ e un OBDD $f$, la robustezza sull'OBDD è calcolata come segue:

1. Si determina il **percorso** che $s$ compie nell'OBDD per raggiungere il nodo `True`, codificandolo come stringa binaria (0 = ramo low, 1 = ramo high), usando l'*Endpoint Universe* per binarizzare i valori delle feature.
2. Si costruisce un **DAG di robustezza** a partire dall'OBDD:
   - Gli archi percorsi da $s$ ricevono peso `0`.
   - Tutti gli altri archi ricevono peso `1`.
   - Il ramo *high* del nodo corrispondente alla **variabile di classe** predetta da $s$ viene rimosso (si forza la foresta a classificare diversamente).
   - Gli archi entranti nel nodo `False` ricevono peso $+\infty$ (il percorso verso `False` non è valido).
3. La robustezza è la **lunghezza del cammino minimo** nel DAG pesato dal nodo radice al nodo `True`.

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

---

## Architettura del sistema e pipeline

```
Input: Random Forest (JSON) + Endpoint Universe (JSON) + Sample (JSON)
        │
        ▼
[1] RandomForestService          ← legge i file JSON dalla cartella results/
        │
        ▼
[2] rf_to_formula_str            ← encoding: RF → formula PSF come stringa
        │
        ▼
[3] from_formula_str / parse_psf ← parsing: stringa PSF → AST (BinaryTree)
        │
        ▼
[4] partial_reduce               ← riduzione parziale: AST → OBDD (se dim ≤ diagram_size)
        │
        ▼
[5] tableau_method               ← Tableau Method: costruisce il TableauTree
        │
        ▼
[6] robustness                   ← calcola la robustezza del sample sul TableauTree
        │
        ▼
Output: valore intero (numero di bit da modificare per cambiare la classificazione)
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
