# rbc-selfplay
Search-Guided Self-Play Learning under Partial Observability: Reconnaissance Blind Chess implementation.

## Setup

```bash
pip install -r requirements.txt

## Run

Open `notebooks/rbc_selfplay.ipynb` with Jupyter or VSCode and run all cells.

## Project structure

- `src/` : core modules (belief, encoder, search, player, training, self-play)
- `notebooks/` : experiments + plots
- `train.py` : training entrypoint (WIP if not fully wired)
- `play_local.py` : local match smoke test
- `sanity_checks.py` : fast consistency checks

## Smoke test (local match)

```bash
python play_local.py


python sanity_checks.py


Commit: `Update README with scripts`

---

## 2) Pulizia: rimuovi `.gitkeep` rimasti dove ci sono file veri
Controlla se esistono ancora:
- `src/.gitkeep` (non serve)
- `notebooks/.gitkeep` (non serve se c’è il notebook)
- altre cartelle non vuote

Se presenti → Delete file → commit `Remove obsolete .gitkeep`

---

## 3) Coerenza nomi: `set_seed` vs `set_seeds`
Nel tuo repo ora:
- nel notebook avevi `set_seeds`
- in `utils.py` hai già `set_seed` nello screenshot

👉 Standardizziamo su **`set_seeds`** (come nel notebook).

Quindi in `src/utils.py` assicurati che la funzione si chiami `set_seeds` e che gli import negli altri file usino quello.

Commit: `Standardize seed function name`

---

## 4) requirements.txt: assicurati che ci siano queste dipendenze minime
Apri `requirements.txt` e assicurati che contenga almeno:


(Se usi tqdm: aggiungi `tqdm`.)

Commit: `Update requirements`

---

## 5) Ultima cosa importante: `train.py` è “WIP”?
Dato che abbiamo rimandato il cablaggio completo di `train_from_data`, metti nel README una nota:

```md
> Note: `train.py` may require wiring `train_from_data()` depending on your current training pipeline.



