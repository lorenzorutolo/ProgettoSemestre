import os
import math
import requests
import csv
import json
from datasets import load_dataset
import numpy as np

# ============================================================
# CONFIGURAZIONI E COSTANTI
# ============================================================
NUM_TEST = 50
RIPETIZIONI_PER_DOMANDA = 3
URL_OLLAMA = 'http://localhost:11434/api/generate'
os.environ["HF_TOKEN"] = "ProgettoSemestre"


# ============================================================
# CLASSE CONTENITORE
# ============================================================
class RisultatiBenchmark:
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.risultati_per_tabella = []
        self.dist_corrette = {}
        self.dist_errate = {}


# ============================================================
# FUNZIONI DI SUPPORTO
# ============================================================
def estrai_prob_da_logprobs(resp_eval: dict) -> tuple:
    p_true = 0.0
    p_false = 0.0
    p_altri = 0.0
    lista_token_grezzi = []

    logprobs = resp_eval.get('logprobs', [])
    if isinstance(logprobs, list) and len(logprobs) > 0:
        candidati_top_k = logprobs[0].get('top_logprobs', [])

        for candidato in candidati_top_k:
            token_originale = candidato.get('token', '')
            testo_candidato = token_originale.strip().lower()

            prob_lineare = math.exp(candidato.get('logprob', -100))
            lista_token_grezzi.append(f"'{token_originale}': {prob_lineare * 100:.8f}%")

            if "true" in testo_candidato:
                p_true += prob_lineare
            elif "false" in testo_candidato:
                p_false += prob_lineare
            else:
                p_altri += prob_lineare

        return p_true, p_false, p_altri, lista_token_grezzi

    return 0.0, 0.0, 0.0, []


def interroga_ollama(payload: dict) -> dict:
    try:
        response = requests.post(URL_OLLAMA, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Errore di connessione a Ollama: {e}")
        return {}


# ============================================================
# LOGICA PRINCIPALE DI BENCHMARK
# ============================================================
def esegui_benchmark() -> RisultatiBenchmark:
    print("Scarico o carico il dataset dalla cache...")
    dataset = load_dataset("google/boolq")
    dati_validazione = dataset['validation'].shuffle()

    risultati = RisultatiBenchmark()

    print(f"\nInizio test su {NUM_TEST} domande (con {RIPETIZIONI_PER_DOMANDA} ripetizioni l'una)...")

    for i in range(NUM_TEST):
        riga = dati_validazione[i]
        testo = riga['passage']
        domanda = riga['question']
        risposta_reale = str(riga['answer']).lower()

        print(f"\nElaborazione Domanda {i + 1}/{NUM_TEST}")

        conteggi = {"true": 0, "false": 0, "altri": 0}
        somma_prob_true = 0.0
        somma_prob_false = 0.0
        somma_prob_altri = 0.0
        distribuzioni_ripetizioni = []

        dati_domanda = {
            "id": i + 1,
            "domanda": domanda,
            "reale": risposta_reale,
            "alternative": [],
            "prob_media_unita": "",
            "generata_dist": "",
            "corretta": False
        }

        domanda_corrente = domanda

        for rep in range(RIPETIZIONI_PER_DOMANDA):
            label = "Original Question " if rep == 0 else "Alternative Question"
            prompt_iniziale = (
                f"You are a strict reading comprehension assistant. Read the following passage carefully.\n"
                f"Your response must be exactly one word: either 'True' or 'False'. Do not include any explanations, introductory text, or punctuation.\n\n"
                f"Passage:\n{testo}\n\n"
                f"Question: {domanda_corrente}\n\n"
                f"Answer:"
            )
            payload_iniziale = {
                'model': 'llama3',
                'prompt': prompt_iniziale,
                'stream': False,
                'options': {'num_predict': 10},
                'raw': False,
                'logprobs': True,
                'top_logprobs': 10
            }

            resp_eval = interroga_ollama(payload_iniziale)
            if not resp_eval:
                continue

            p_true, p_false, p_altri, token_grezzi = estrai_prob_da_logprobs(resp_eval)

            distribuzioni_ripetizioni.append(f"({p_true:.8f}, {p_false:.8f}, {p_altri:.8f})")

            # Accumula le prob della singola ripetizione
            somma_prob_true += p_true
            somma_prob_false += p_false
            somma_prob_altri += p_altri

            # Classificazione basata sulla singola ripetizione (non sulla somma cumulativa)
            if p_true > p_false:
                testo_generato = "true"
                conteggi["true"] += 1
            elif p_false > p_true:
                testo_generato = "false"
                conteggi["false"] += 1
            else:
                testo_generato = "altro"
                conteggi["altri"] += 1

            print(f"  - {label} ({rep + 1}): {domanda_corrente}")
            if token_grezzi:
                print(f"Vettore Token Rilevati (Top 10): [{', '.join(token_grezzi)}]")

            dati_domanda["alternative"].append({
                "domanda_alt": domanda_corrente,
                "risposta_pulita": testo_generato,
                "prob_unita_alt": f"T:{(p_true * 100):.8f}% F:{(p_false * 100):.8f}% O:{(p_altri * 100):.8f}%",
                "p_true_raw": p_true,
                "p_false_raw": p_false,
                "p_altri_raw": p_altri  

            })

            prompt_perturbazione = (
                f"You are an expert linguistic assistant. Your only task is to paraphrase the given question.\n"
                f"Rewrite the question using different words or sentence structure, but keep the exact same logical meaning and intent.\n"
                f"Your response must contain ONLY the new paraphrased question. Do not include any introductory phrases, explanations, or answers.\n\n"
                f"Question: {domanda_corrente}\n\n"
                f"Paraphrased Question:"
            )
            payload_perturbazione = {
                'model': 'llama3',
                'prompt': prompt_perturbazione,
                'stream': False,
                'options': {'num_predict': 10},
                'raw': False
            }
            resp_pert = interroga_ollama(payload_perturbazione)
            domanda_corrente = resp_pert.get('response', '').strip() if resp_pert else domanda_corrente

        print(f"  -> Vettore distribuzioni (T, F, A): [{', '.join(distribuzioni_ripetizioni)}]")

        # Decisione finale fuori dal loop, basata sulle somme cumulative delle prob
        if somma_prob_true > somma_prob_false:
            risposta_scelta_modello = "true"
        elif somma_prob_false > somma_prob_true:
            risposta_scelta_modello = "false"
        else:
            risposta_scelta_modello = "pareggio"

        valori = sorted([conteggi["true"], conteggi["false"]], reverse=True)
        chiave_distribuzione = f"{valori[0]}-{valori[1]}"
        esito_corretto = (risposta_scelta_modello == risposta_reale)

        if esito_corretto:
            risultati.dist_corrette[chiave_distribuzione] = risultati.dist_corrette.get(chiave_distribuzione, 0) + 1
            if risposta_reale == "true":
                risultati.tp += 1
            else:
                risultati.tn += 1
        else:
            risultati.dist_errate[chiave_distribuzione] = risultati.dist_errate.get(chiave_distribuzione, 0) + 1
            if risposta_reale == "false":
                risultati.fp += 1
            else:
                risultati.fn += 1

        perc_true_avg = (somma_prob_true / RIPETIZIONI_PER_DOMANDA) * 100
        perc_false_avg = (somma_prob_false / RIPETIZIONI_PER_DOMANDA) * 100
        perc_altri_avg = (somma_prob_altri / RIPETIZIONI_PER_DOMANDA) * 100

        dati_domanda["prob_media_unita"] = f"T:{perc_true_avg:.8f}% F:{perc_false_avg:.8f}% A:{perc_altri_avg:.8f}%"
        dati_domanda["generata_dist"] = chiave_distribuzione
        dati_domanda["corretta"] = esito_corretto

        risultati.risultati_per_tabella.append(dati_domanda)

    return risultati
# ============================================================
# SALVATAGGIO IN CSV
# ============================================================
def salva_su_csv(risultati: RisultatiBenchmark, filename="risultati_benchmark.csv"):
    print(f"\nSalvataggio dei dati nel file '{filename}'...")
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Intestazione del CSV
        writer.writerow(["id", "domanda", "reale", "alternative_json", "prob_media_unita", "generata_dist", "corretta"])

        for riga in risultati.risultati_per_tabella:
            # Salviamo il dictionary annidato 'alternative' come stringa JSON
            alternative_json = json.dumps(riga["alternative"])
            writer.writerow([
                riga["id"],
                riga["domanda"],
                riga["reale"],
                alternative_json,
                riga["prob_media_unita"],
                riga["generata_dist"],
                riga["corretta"]
            ])
    print("Salvataggio completato!")


if __name__ == "__main__":
    dati_benchmark = esegui_benchmark()
    salva_su_csv(dati_benchmark)