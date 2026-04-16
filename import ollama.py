import os
import time
import math
import requests
import tkinter as tk
from tkinter import ttk

import seaborn as sns
from datasets import load_dataset
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MultipleLocator

# ============================================================
# CONFIGURAZIONI E COSTANTI
# ============================================================
NUM_TEST = 10
TEMPERATURA = 0.9
RIPETIZIONI_PER_DOMANDA = 10
URL_OLLAMA = 'http://localhost:11434/api/generate'
os.environ["HF_TOKEN"] = "ProgettoSemestre"


# ============================================================
# CLASSE CONTENITORE PER RAGGRUPPARE I CONTENUTI DEL BENCHMARK
# ============================================================
class RisultatiBenchmark:
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.entropia_corrette = []
        self.entropia_errate = []
        self.risultati_per_tabella = []
        self.dist_corrette = {}
        self.dist_errate = {}


# ============================================================
# CALCOLA L'ENTROPIA DI SHANNON DI UNA DISTRIBUZIONE BINARIA
# ============================================================
def entropia_binaria(p_true: float, p_false: float) -> float:
    h = 0.0
    for p in (p_true, p_false):
        if p > 0:
            h -= p * math.log2(p)
    return h


# ============================================================
# WRAPPER PER GESTIRE LE RICHIESTE HTTP A OLLAMA
# ============================================================
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
    start = time.time()
    print("Scarico o carico il dataset dalla cache...")
    dataset = load_dataset("google/boolq")
    dati_validazione = dataset['validation']

    risultati = RisultatiBenchmark()

    print(f"\nInizio test su {NUM_TEST} domande (con {RIPETIZIONI_PER_DOMANDA} ripetizioni l'una)...")

    for i in range(NUM_TEST):
        riga = dati_validazione[i]
        testo = riga['passage']
        domanda = riga['question']
        risposta_reale = str(riga['answer']).lower()

        print(f"\nElaborazione Domanda {i + 1}/{NUM_TEST}")
        print(f"- Domanda Originale: ", domanda)

        conteggi = {"true": 0, "false": 0, "errore": 0}
        running_true = 0
        running_false = 0
        entropie_cumulative = []

        somma_prob_true = 0.0
        somma_prob_false = 0.0
        conteggio_prob_valide = 0

        distribuzioni_ripetizioni = []
        dettagli_alternative = []

        prompt_alternative_text = (
            f"You are an expert linguistic assistant. Your only task is to paraphrase the given question.\n"
            f"Rewrite the question using different words or sentence structure, but keep the exact same logical meaning and intent.\n"
            f"Use the provided 'Passage' ONLY as context to understand the question. DO NOT answer the question.\n"
            f"Your response must contain ONLY the new paraphrased question. Do not include any introductory phrases, explanations, or answers.\n\n"
            f"Passage:\n{testo}\n\n"
            f"Original Question: {domanda}\n\n"
            f"Paraphrased Question:"
        )

        payload_alternative = {
            'model': 'llama3',
            'prompt': prompt_alternative_text,
            'stream': False,
            'options': {'num_predict': 100, 'temperature': 0.8},
            'raw': False
        }

        for rep in range(RIPETIZIONI_PER_DOMANDA):
            # ── 1. Generazione della domanda alternativa ──
            resp_alt = interroga_ollama(payload_alternative)
            domanda_alt = resp_alt.get('response', '').strip() if resp_alt else domanda
            print(f"  - Alternative Question ({rep + 1}): {domanda_alt}")

            # ── 2. Risposta (Valutazione True/False) della domanda alternativa ──
            prompt_completo = (
                f"You are a strict reading comprehension assistant. Read the following passage carefully.\n"
                f"Your response must be exactly one word: either 'True' or 'False'. Do not include any explanations, introductory text, or punctuation.\n\n"
                f"Passage:\n{testo}\n\n"
                f"Question: {domanda_alt}\n\n"
                f"Answer:"
            )

            payload_eval = {
                'model': 'llama3',
                'prompt': prompt_completo,
                'stream': False,
                'options': {'num_predict': 100, 'temperature': TEMPERATURA},
                'raw': False,
                'logprobs': True,
                'top_logprobs': 5
            }

            resp_eval = interroga_ollama(payload_eval)
            if not resp_eval:
                continue  # Salta in caso di errore di connessione

            testo_generato = resp_eval.get('response', '').strip().lower()
            risposta_pulita = testo_generato.replace(".", "").replace(",", "")

            if "true" in risposta_pulita:
                conteggi["true"] += 1
                running_true += 1
            elif "false" in risposta_pulita:
                conteggi["false"] += 1
                running_false += 1
            else:
                conteggi["errore"] += 1

            k_valide = running_true + running_false
            if k_valide > 0:
                entropie_cumulative.append(entropia_binaria(running_true / k_valide, running_false / k_valide))

            # ── Estrazione logprobs per la singola alternativa ──
            probabilita_grezza_true = 0.0
            probabilita_grezza_false = 0.0
            prob_norm_t_str = "0.50"
            prob_norm_f_str = "0.50"
            h_singola = 0.0

            logprobs = resp_eval.get('logprobs', [])
            if isinstance(logprobs, list) and len(logprobs) > 0:
                candidati_top_k = logprobs[0].get('top_logprobs', [])

                for candidato in candidati_top_k:
                    testo_candidato = candidato.get('token', '').strip().lower()
                    prob_lineare = math.exp(candidato.get('logprob', -100))

                    if "true" in testo_candidato:
                        probabilita_grezza_true += prob_lineare
                    elif "false" in testo_candidato:
                        probabilita_grezza_false += prob_lineare

                somma_rep = probabilita_grezza_true + probabilita_grezza_false
                if somma_rep > 0:
                    p_norm_t = probabilita_grezza_true / somma_rep
                    p_norm_f = probabilita_grezza_false / somma_rep

                    somma_prob_true += p_norm_t
                    somma_prob_false += p_norm_f
                    conteggio_prob_valide += 1

                    distribuzioni_ripetizioni.append(f"({p_norm_t:.2f}, {p_norm_f:.2f})")
                    prob_norm_t_str = f"{p_norm_t:.2f}"
                    prob_norm_f_str = f"{p_norm_f:.2f}"
                    h_singola = entropia_binaria(p_norm_t, p_norm_f)
                else:
                    distribuzioni_ripetizioni.append("(0.50, 0.50)")
                    h_singola = entropia_binaria(0.5, 0.5)

            # Salvataggio dettagli
            dettagli_alternative.append({
                "domanda_alt": domanda_alt,
                "risposta_pulita": risposta_pulita,
                "prob_t": prob_norm_t_str,
                "prob_f": prob_norm_f_str,
                "h_singola": h_singola
            })

        print(f"  -> Vettore distribuzioni (T, F): [{', '.join(distribuzioni_ripetizioni)}]")

        # ── Aggregazione Dati Domanda ──
        h_min = min(entropie_cumulative) if entropie_cumulative else 0.0
        h_max = max(entropie_cumulative) if entropie_cumulative else 0.0

        tot_valide = running_true + running_false
        if tot_valide > 0:
            h_media_val = entropia_binaria(running_true / tot_valide, running_false / tot_valide)
        else:
            h_media_val = entropia_binaria(0.5, 0.5)

        # Maggioranza
        if conteggi["true"] > conteggi["false"]:
            risposta_scelta_modello = "true"
        elif conteggi["false"] > conteggi["true"]:
            risposta_scelta_modello = "false"
        else:
            risposta_scelta_modello = "pareggio"

        valori = sorted([conteggi["true"], conteggi["false"]], reverse=True)
        chiave_distribuzione = f"{valori[0]}-{valori[1]}"

        esito_corretto = (risposta_scelta_modello == risposta_reale)

        # Salvataggio Esiti su Oggetto Risultati
        if esito_corretto:
            risultati.dist_corrette[chiave_distribuzione] = risultati.dist_corrette.get(chiave_distribuzione, 0) + 1
            risultati.entropia_corrette.append(h_media_val)
            if risposta_reale == "true":
                risultati.tp += 1
            else:
                risultati.tn += 1
        else:
            risultati.dist_errate[chiave_distribuzione] = risultati.dist_errate.get(chiave_distribuzione, 0) + 1
            risultati.entropia_errate.append(h_media_val)
            if risposta_reale == "false":
                risultati.fp += 1
            else:
                risultati.fn += 1

        perc_true_avg = (somma_prob_true / conteggio_prob_valide * 100) if conteggio_prob_valide > 0 else 50.0
        perc_false_avg = (somma_prob_false / conteggio_prob_valide * 100) if conteggio_prob_valide > 0 else 50.0

        risultati.risultati_per_tabella.append({
            "domanda": domanda,
            "prob_media": f"T:{perc_true_avg:.1f}% F:{perc_false_avg:.1f}%",
            "generata_dist": chiave_distribuzione,
            "reale": risposta_reale,
            "corretta": esito_corretto,
            "h_min": h_min,
            "h_max": h_max,
            "h_media": h_media_val,
            "alternative": dettagli_alternative
        })

    end = time.time()
    print(f"\nTest completato in {end - start:.1f} secondi! Genero l'interfaccia...")
    return risultati


# ============================================================
# INTERFACCIA GRAFICA
# ============================================================
def mostra_interfaccia_completa(res: RisultatiBenchmark):
    finestra = tk.Tk()
    finestra.title(f"Report Benchmark ({RIPETIZIONI_PER_DOMANDA} Ripetizioni su {NUM_TEST} Domande)")
    finestra.geometry("1100x650")

    notebook = ttk.Notebook(finestra)
    notebook.pack(fill='both', expand=True, padx=10, pady=10)

    # ── SCHEDA 1: Tabella ad Albero e Matrice ──
    tab1 = ttk.Frame(notebook)
    notebook.add(tab1, text="Dati e Matrice")

    totale_corrette = res.tp + res.tn
    percentuale_corrette = (totale_corrette / NUM_TEST) * 100 if NUM_TEST > 0 else 0
    tk.Label(tab1, text=f"Risultati (Maggioranza): {totale_corrette}/{NUM_TEST} Corrette ({percentuale_corrette:.1f}%)",
             font=("Helvetica", 14, "bold")).pack(pady=10)

    # Matrice di confusione
    frame_matrice = tk.Frame(tab1)
    frame_matrice.pack(side=tk.BOTTOM, pady=10)
    tk.Label(frame_matrice, text="Matrice di Confusione", font=("Helvetica", 12, "bold")).grid(row=0, column=0,
                                                                                               columnspan=3, pady=5)
    tk.Label(frame_matrice, text="Modello: True").grid(row=1, column=1)
    tk.Label(frame_matrice, text="Modello: False").grid(row=1, column=2)
    tk.Label(frame_matrice, text="Realtà: True").grid(row=2, column=0)
    tk.Label(frame_matrice, text=f"TP\n{res.tp}", bg="#c6efce", width=10, height=2, relief="groove").grid(row=2,
                                                                                                          column=1)
    tk.Label(frame_matrice, text=f"FN\n{res.fn}", bg="#ffc7ce", width=10, height=2, relief="groove").grid(row=2,
                                                                                                          column=2)
    tk.Label(frame_matrice, text="Realtà: False").grid(row=3, column=0)
    tk.Label(frame_matrice, text=f"FP\n{res.fp}", bg="#ffc7ce", width=10, height=2, relief="groove").grid(row=3,
                                                                                                          column=1)
    tk.Label(frame_matrice, text=f"TN\n{res.tn}", bg="#c6efce", width=10, height=2, relief="groove").grid(row=3,
                                                                                                          column=2)

    # Tabella
    frame_tabella = tk.Frame(tab1)
    frame_tabella.pack(side=tk.TOP, fill="both", expand=True, padx=10, pady=5)

    colonne = ("Domanda", "ProbMedia", "Distribuzione", "Reale", "H_min", "H_max", "H_media")
    tabella = ttk.Treeview(frame_tabella, columns=colonne, show="tree headings")

    tabella.heading("#0", text="");
    tabella.column("#0", width=40, stretch=tk.NO, anchor="center")
    tabella.heading("Domanda", text="Domanda Originale / Varianti");
    tabella.column("Domanda", width=250)
    tabella.heading("ProbMedia", text=f"Media Prob. [{RIPETIZIONI_PER_DOMANDA} Rip]");
    tabella.column("ProbMedia", width=135, anchor="center")
    tabella.heading("Distribuzione", text="Dist. [Rip.]");
    tabella.column("Distribuzione", width=70, anchor="center")
    tabella.heading("Reale", text="Reale / Gen.");
    tabella.column("Reale", width=80, anchor="center")
    tabella.heading("H_min", text="H min");
    tabella.column("H_min", width=60, anchor="center")
    tabella.heading("H_max", text="H max");
    tabella.column("H_max", width=60, anchor="center")
    tabella.heading("H_media", text="H media (bit)");
    tabella.column("H_media", width=90, anchor="center")

    scrollbar = ttk.Scrollbar(frame_tabella, orient=tk.VERTICAL, command=tabella.yview)
    tabella.configure(yscroll=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    tabella.pack(side=tk.LEFT, fill="both", expand=True)

    tabella.tag_configure("verde", background="#c6efce", foreground="#006100")
    tabella.tag_configure("rosso", background="#ffc7ce", foreground="#9c0006")
    tabella.tag_configure("figlio", background="#f0f8ff", foreground="#333333")

    for riga in res.risultati_per_tabella:
        colore = "verde" if riga["corretta"] else "rosso"
        padre_id = tabella.insert("", tk.END, text="", values=(
            riga["domanda"][:60] + "...", riga["prob_media"], riga["generata_dist"], riga["reale"].upper(),
            f"{riga['h_min']:.3f}", f"{riga['h_max']:.3f}", f"{riga['h_media']:.2f}"
        ), tags=(colore,))

        for i, alt in enumerate(riga["alternative"]):
            tabella.insert(padre_id, tk.END, text=f"{i + 1}.", values=(
                " ↳ " + alt["domanda_alt"][:65] + "...", f"T:{alt['prob_t']} F:{alt['prob_f']}", "-",
                alt["risposta_pulita"].upper(), "-", "-", f"{alt['h_singola']:.2f}"
            ), tags=("figlio",))

    # ── SCHEDA 2: GRAFICO ENTROPIA ──
    tab2 = ttk.Frame(notebook)
    notebook.add(tab2, text="Grafico Entropia")
    fig = Figure(figsize=(7, 4), dpi=100)
    ax = fig.add_subplot(111)

    # Ottimizzazione popolamento dati grafico
    dati_plot = res.entropia_corrette + res.entropia_errate + res.entropia_corrette + res.entropia_errate
    etichette = (["Corrette"] * len(res.entropia_corrette) +
                 ["Errate"] * len(res.entropia_errate) +
                 ["Aggregato"] * (len(res.entropia_corrette) + len(res.entropia_errate)))

    if dati_plot:
        sns.boxplot(x=etichette, y=dati_plot, hue=etichette,
                    palette={"Corrette": "#c6efce", "Errate": "#ffc7ce", "Aggregato": "#e0e0e0"},
                    order=["Corrette", "Errate", "Aggregato"], ax=ax, legend=False, fliersize=0)
        sns.stripplot(x=etichette, y=dati_plot, hue=etichette,
                      palette={"Corrette": "#006100", "Errate": "#9c0006", "Aggregato": "#333333"},
                      order=["Corrette", "Errate", "Aggregato"], ax=ax, legend=False, jitter=True, alpha=0.7, size=5)

    ax.set_title("Entropia Media del Modello (bit)")
    ax.set_ylabel("Entropia H (bit)  [0 = certo, 1 = massima incertezza]")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    FigureCanvasTkAgg(fig, master=tab2).get_tk_widget().pack(fill='both', expand=True)

    # ── PREPARAZIONE DATI SCHEDA 3 E 4 ──
    categorie_x = [f"{k}-{RIPETIZIONI_PER_DOMANDA - k}" for k in
                   range(RIPETIZIONI_PER_DOMANDA, math.ceil(RIPETIZIONI_PER_DOMANDA / 2) - 1, -1)]
    chiavi_extra = sorted(set(res.dist_corrette.keys()).union(set(res.dist_errate.keys())), reverse=True)
    for key in chiavi_extra:
        if key not in categorie_x:
            categorie_x.append(key)

    domande_giuste = [res.dist_corrette.get(cat, 0) for cat in categorie_x]
    domande_sbagliate = [res.dist_errate.get(cat, 0) for cat in categorie_x]

    # ── SCHEDA 3: CONTEGGIO DOMANDE ──
    tab3 = ttk.Frame(notebook)
    notebook.add(tab3, text="Conteggio Domande")
    fig3 = Figure(figsize=(6, 4), dpi=100)
    ax3 = fig3.add_subplot(111)
    ax3.bar(categorie_x, domande_giuste, color='#4CAF50', label='Domande Corrette')
    ax3.bar(categorie_x, domande_sbagliate, bottom=domande_giuste, color='#F44336', label='Domande Errate')
    ax3.set_title("Conteggio Domande Corrette/Errate per Distribuzione")
    ax3.set_xlabel("Combinazione (Maggioranza - Minoranza)")
    ax3.set_ylabel("Numero di Domande Totali")
    ax3.legend()
    ax3.yaxis.set_major_locator(MultipleLocator(max(1, NUM_TEST // 10)))
    ax3.grid(axis='y', which='major', linestyle='-', linewidth=0.8, alpha=0.7)
    FigureCanvasTkAgg(fig3, master=tab3).get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

    # ── SCHEDA 4: DIAGRAMMA CALIBRAZIONE ──
    tab4 = ttk.Frame(notebook)
    notebook.add(tab4, text="Diagramma di Calibrazione (%)")

    perc_giuste = [(g / (g + s) * 100) if (g + s) > 0 else 0 for g, s in zip(domande_giuste, domande_sbagliate)]
    perc_sbagliate = [(s / (g + s) * 100) if (g + s) > 0 else 0 for g, s in zip(domande_giuste, domande_sbagliate)]

    fig4 = Figure(figsize=(6, 4), dpi=100)
    ax4 = fig4.add_subplot(111)
    x = range(len(categorie_x))
    width = 0.35

    bars_giuste = ax4.bar([i - width / 2 for i in x], perc_giuste, width, color='#4CAF50', label='Domande Corrette (%)')
    bars_errate = ax4.bar([i + width / 2 for i in x], perc_sbagliate, width, color='#F44336',
                          label='Domande Errate (%)')

    def annota_barre(bars, color):
        for bar in bars:
            height = bar.get_height()
            ax4.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                         fontsize=9, color=color, fontweight='bold')

    annota_barre(bars_giuste, '#006100')
    annota_barre(bars_errate, '#9c0006')

    ax4.set_xticks(x)
    ax4.set_xticklabels(categorie_x)
    ax4.set_title("Diagramma di Calibrazione: Tasso Corrette/Errate per Distribuzione")
    ax4.set_xlabel("Combinazione (Maggioranza - Minoranza)")
    ax4.set_ylabel("Percentuale di Accuratezza/Errore (%)")
    ax4.set_ylim(0, 115)
    ax4.legend(loc='upper right')
    ax4.yaxis.set_major_locator(MultipleLocator(10))
    ax4.grid(axis='y', linestyle='--', alpha=0.7)

    FigureCanvasTkAgg(fig4, master=tab4).get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

    finestra.mainloop()


# ============================================================
# AVVIO DELLO SCRIPT
# ============================================================
if __name__ == "__main__":
    dati_benchmark = esegui_benchmark()
    mostra_interfaccia_completa(dati_benchmark)