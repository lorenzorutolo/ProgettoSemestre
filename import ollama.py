import os
import time
import math
import requests
import tkinter as tk
from tkinter import ttk

from datasets import load_dataset
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MultipleLocator

# ============================================================
# CONFIGURAZIONI E COSTANTI
# ============================================================
NUM_TEST = 50
TEMPERATURA = 1.2
RIPETIZIONI_PER_DOMANDA = 8
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
        self.risultati_per_tabella = []
        self.dist_corrette = {}
        self.dist_errate = {}


# ============================================================
# ESTRAE LE PROBABILITÀ GREZZE
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

    return 0.0, 0.0, 0.0, False, []


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
            # prompt iniziale dove chiediamo di rispondere true e false
            prompt_iniziale = (
                f"You are a strict reading comprehension assistant. Read the following passage carefully.\n"
                f"Your response must be exactly one word: either 'True' or 'False'. Do not include any explanations, introductory text, or punctuation.\n\n"
                f"Passage:\n{testo}\n\n"
                f"Question: {domanda_corrente}\n\n"
                f"Answer:"
            )
            # payload iniziale
            payload_iniziale = {
                'model': 'llama3',
                'prompt': prompt_iniziale,
                'stream': False,
                'options': {'num_predict': 100, 'temperature': TEMPERATURA},
                'raw': False,
                'logprobs': True,
                'top_logprobs': 10
            }
            # resp eval contiene la risposta del modello completa che comprende nome modello, token utilizzati, temperatura, contesto (illeggibile così com'è)
            resp_eval = interroga_ollama(payload_iniziale)
            if not resp_eval:
                continue
            # prendiamo la risposta alla chiave "response" e se non esiste mettimao una stringa vuota
            testo_generato = resp_eval.get('response', '').strip().lower()
            # togliamo punti, virgola parentesi ecc..
            risposta_pulita = testo_generato.replace(".", "").replace(",", "").replace("(", "").replace("_", "")
            print("\n risposa pulita", risposta_pulita)

            if "true" in risposta_pulita:
                conteggi["true"] += 1
            elif "false" in risposta_pulita:
                conteggi["false"] += 1
            else:
                conteggi["altri"] += 1

            p_true, p_false, p_altri, token_grezzi = estrai_prob_da_logprobs(resp_eval)

            distribuzioni_ripetizioni.append(f"({p_true:.8f}, {p_false:.8f}, {p_altri:.8f})")

            somma_prob_true += p_true
            somma_prob_false += p_false
            somma_prob_altri += p_altri

            print(f"  - {label} ({rep + 1}): {domanda_corrente}")
            if token_grezzi:
                print(f"Vettore Token Rilevati (Top 10): [{', '.join(token_grezzi)}]")

            # Iniettiamo i dati nella lista della nostra variabile principale
            dati_domanda["alternative"].append({
                "domanda_alt": domanda_corrente,
                "risposta_pulita": risposta_pulita,
                "prob_unita_alt": f"T:{(p_true * 100):.8f}% F:{(p_false * 100):.8f}% O:{(p_altri * 100):.8f}%",
                "p_true_raw": p_true,
                "p_false_raw": p_false
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
                'options': {'num_predict': 100, 'temperature': 0.8},
                'raw': False
            }
            resp_pert = interroga_ollama(payload_perturbazione)
            domanda_corrente = resp_pert.get('response', '').strip() if resp_pert else domanda_corrente

        print(f"  -> Vettore distribuzioni (T, F, A): [{', '.join(distribuzioni_ripetizioni)}]")

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
        # matrice di confusione
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
        # medie delle distribuzioni
        perc_true_avg = (somma_prob_true / RIPETIZIONI_PER_DOMANDA) * 100
        perc_false_avg = (somma_prob_false / RIPETIZIONI_PER_DOMANDA) * 100
        perc_altri_avg = (somma_prob_altri / RIPETIZIONI_PER_DOMANDA) * 100

        dati_domanda["prob_media_unita"] = f"T:{perc_true_avg:.8f}% F:{perc_false_avg:.8f}% A:{perc_altri_avg:.8f}%"
        dati_domanda["generata_dist"] = chiave_distribuzione
        dati_domanda["corretta"] = esito_corretto

        # Salviamo l'intero pacchetto nel contenitore finale
        risultati.risultati_per_tabella.append(dati_domanda)

    return risultati


# ============================================================
# ANALYZER: CALCOLO STATISTICHE ENTROPIA
# ============================================================
def analyzer(risultati: RisultatiBenchmark) -> RisultatiBenchmark:
    for riga in risultati.risultati_per_tabella:
        entropie = []
        somma_norm_t = 0.0
        somma_norm_f = 0.0

        for alt in riga["alternative"]:
            p_t = alt.get("p_true_raw", 0.0)
            p_f = alt.get("p_false_raw", 0.0)

            # Normalizzazione True/False rispetto allo spazio T/F
            somma_parziale = p_t + p_f
            if somma_parziale > 0:
                n_t = p_t / somma_parziale
                n_f = p_f / somma_parziale
            else:
                n_t, n_f = 0.0, 0.0

            # Calcolo entropia della singola distribuzione normalizzata
            ent_singola = 0.0
            if n_t > 0: ent_singola -= n_t * math.log2(n_t)
            if n_f > 0: ent_singola -= n_f * math.log2(n_f)

            entropie.append(ent_singola)
            somma_norm_t += n_t
            somma_norm_f += n_f
        # k = numero di ripetizioni
        k = len(riga["alternative"])
        if k > 0:
            # entropia minima / massima
            riga["min_ent"] = min(entropie)
            riga["max_ent"] = max(entropie)

            # Entropia della media delle Distribuzioni
            avg_n_t = somma_norm_t / k
            avg_n_f = somma_norm_f / k

            ent_media = 0.0
            if avg_n_t > 0: ent_media -= avg_n_t * math.log2(avg_n_t)
            if avg_n_f > 0: ent_media -= avg_n_f * math.log2(avg_n_f)

            riga["avg_ent"] = ent_media
        else:
            riga["min_ent"] = 0.0
            riga["max_ent"] = 0.0
            riga["avg_ent"] = 0.0

    return risultati


# ============================================================
# INTERFACCIA GRAFICA (AGGIORNATA CON I NUOVI PLOT)
# ============================================================
def mostra_interfaccia_completa(res: RisultatiBenchmark):
    finestra = tk.Tk()
    finestra.title(f"Report Benchmark ({RIPETIZIONI_PER_DOMANDA} Ripetizioni su {NUM_TEST} Domande)")
    finestra.geometry("1350x650")

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

    colonne = ("ID", "Domanda", "Probabilita", "Distribuzione", "Reale", "MinEnt", "MaxEnt", "AvgEnt")
    tabella = ttk.Treeview(frame_tabella, columns=colonne, show="tree headings")

    tabella.heading("#0", text="")
    tabella.column("#0", width=40, stretch=tk.NO, anchor="center")
    tabella.heading("ID", text="N°")
    tabella.column("ID", width=40, anchor="center")
    tabella.heading("Domanda", text="Domanda Originale / Varianti")
    tabella.column("Domanda", width=250)
    tabella.heading("Probabilita", text=f"Medie (%T / %F / %O) [{RIPETIZIONI_PER_DOMANDA} Rip]")
    tabella.column("Probabilita", width=380, anchor="center")
    tabella.heading("Distribuzione", text="Dist. [Rip.]")
    tabella.column("Distribuzione", width=70, anchor="center")
    tabella.heading("Reale", text="Reale")
    tabella.column("Reale", width=80, anchor="center")
    tabella.heading("MinEnt", text="MinEnt")
    tabella.column("MinEnt", width=60, anchor="center")
    tabella.heading("MaxEnt", text="MaxEnt")
    tabella.column("MaxEnt", width=60, anchor="center")
    tabella.heading("AvgEnt", text="AvgEnt")
    tabella.column("AvgEnt", width=60, anchor="center")

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
            riga["id"],
            riga["domanda"][:60] + "...",
            riga["prob_media_unita"],
            riga["generata_dist"],
            riga["reale"].upper(),
            f"{riga.get('min_ent', 0):.3f}",
            f"{riga.get('max_ent', 0):.3f}",
            f"{riga.get('avg_ent', 0):.3f}"
        ), tags=(colore,))

        for i, alt in enumerate(riga["alternative"]):
            tabella.insert(padre_id, tk.END, text=f"{i + 1}.", values=(
                "",
                " ↳ " + alt["domanda_alt"][:65] + "...",
                alt['prob_unita_alt'],
                "-",
                alt["risposta_pulita"].upper(),
                "", "", ""
            ), tags=("figlio",))

    # ── SCHEDA 3 E 4 (I TUOI PLOT ORIGINALI) ──
    categorie_x = [f"{k}-{RIPETIZIONI_PER_DOMANDA - k}" for k in
                   range(RIPETIZIONI_PER_DOMANDA, math.ceil(RIPETIZIONI_PER_DOMANDA / 2) - 1, -1)]
    chiavi_extra = sorted(set(res.dist_corrette.keys()).union(set(res.dist_errate.keys())), reverse=True)
    for key in chiavi_extra:
        if key not in categorie_x:
            categorie_x.append(key)

    domande_giuste = [res.dist_corrette.get(cat, 0) for cat in categorie_x]
    domande_sbagliate = [res.dist_errate.get(cat, 0) for cat in categorie_x]

    tab3 = ttk.Frame(notebook)
    notebook.add(tab3, text="Conteggio per Distrib.")
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

    tab4 = ttk.Frame(notebook)
    notebook.add(tab4, text="Calibrazione per Distrib.")
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

    # ============================================================
    # NUOVI PLOT AGGIUNTI (MAX ENTROPIA E AVG ENTROPIA)
    # ============================================================

    # Prepariamo le liste filtrando per risposte giuste e sbagliate
    max_ent_corrette = [r.get("max_ent", 0.0) for r in res.risultati_per_tabella if r["corretta"]]
    max_ent_errate = [r.get("max_ent", 0.0) for r in res.risultati_per_tabella if not r["corretta"]]

    avg_ent_corrette = [r.get("avg_ent", 0.0) for r in res.risultati_per_tabella if r["corretta"]]
    avg_ent_errate = [r.get("avg_ent", 0.0) for r in res.risultati_per_tabella if not r["corretta"]]

    # ── SCHEDA 5: BOXPLOT MAX ENTROPIA ──
    tab5 = ttk.Frame(notebook)
    notebook.add(tab5, text="Boxplot MaxEnt")
    fig5 = Figure(figsize=(6, 4), dpi=100)
    ax5 = fig5.add_subplot(111)

    # Crea il boxplot. Se una lista è vuota passiamo uno 0 temporaneo per non far crashare matplotlib
    dati_max = [max_ent_corrette if max_ent_corrette else [0.0],
                max_ent_errate if max_ent_errate else [0.0]]

    bplot1 = ax5.boxplot(dati_max, labels=['Corrette', 'Errate'], patch_artist=True)
    colors = ['#4CAF50', '#F44336']  # Verde e Rosso
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax5.set_title("Distribuzione MaxEnt per Domande Corrette ed Errate")
    ax5.set_ylabel("Entropia Massima (MaxEnt) [0.0 - 1.0]")
    ax5.set_ylim(-0.05, 1.05)
    ax5.grid(axis='y', linestyle='--', alpha=0.7)
    FigureCanvasTkAgg(fig5, master=tab5).get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

    # ── SCHEDA 6: BOXPLOT AVG ENTROPIA ──
    tab6 = ttk.Frame(notebook)
    notebook.add(tab6, text="Boxplot AvgEnt")
    fig6 = Figure(figsize=(6, 4), dpi=100)
    ax6 = fig6.add_subplot(111)

    dati_avg = [avg_ent_corrette if avg_ent_corrette else [0.0],
                avg_ent_errate if avg_ent_errate else [0.0]]

    bplot2 = ax6.boxplot(dati_avg, labels=['Corrette', 'Errate'], patch_artist=True)
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax6.set_title("Distribuzione AvgEnt per Domande Corrette ed Errate")
    ax6.set_ylabel("Entropia Media (AvgEnt) [0.0 - 1.0]")
    ax6.set_ylim(-0.05, 1.05)
    ax6.grid(axis='y', linestyle='--', alpha=0.7)
    FigureCanvasTkAgg(fig6, master=tab6).get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

    # ── SCHEDA 7: ACCURATEZZA PER LIVELLI DI ENTROPIA (BAR CHART) ──
    tab7 = ttk.Frame(notebook)
    notebook.add(tab7, text="Accuratezza vs Entropia")
    fig7 = Figure(figsize=(6, 4), dpi=100)
    ax7 = fig7.add_subplot(111)

    # Dividiamo i dati dell'AvgEnt (che misura l'incertezza globale) in 3 fasce
    bassa = [r["corretta"] for r in res.risultati_per_tabella if r.get("avg_ent", 0.0) <= 0.3]
    media = [r["corretta"] for r in res.risultati_per_tabella if 0.3 < r.get("avg_ent", 0.0) <= 0.7]
    alta = [r["corretta"] for r in res.risultati_per_tabella if r.get("avg_ent", 0.0) > 0.7]

    # Calcoliamo l'accuratezza % per ogni fascia. Se una fascia è vuota, mettiamo 0
    acc_bassa = (sum(bassa) / len(bassa) * 100) if len(bassa) > 0 else 0
    acc_media = (sum(media) / len(media) * 100) if len(media) > 0 else 0
    acc_alta = (sum(alta) / len(alta) * 100) if len(alta) > 0 else 0

    etichette_fasce = ['Bassa\n(0.0 - 0.3)', 'Media\n(0.3 - 0.7)', 'Alta\n(0.7 - 1.0)']
    valori_accuratezza = [acc_bassa, acc_media, acc_alta]

    barre_calibrazione = ax7.bar(etichette_fasce, valori_accuratezza, color='#2196F3', alpha=0.8)

    # Aggiungiamo le etichette con i valori sopra ogni barra
    for bar in barre_calibrazione:
        altezza = bar.get_height()
        ax7.annotate(f'{altezza:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, altezza),
                     xytext=(0, 3),  # offset verticale di 3 punti
                     textcoords="offset points",
                     ha='center', va='bottom', fontweight='bold')

    ax7.set_title("Accuratezza del Modello per Livelli di Entropia (AvgEnt)")
    ax7.set_xlabel("Fascia di Entropia")
    ax7.set_ylabel("Accuratezza (%)")
    ax7.set_ylim(0, 115)
    ax7.grid(axis='y', linestyle='--', alpha=0.7)
    FigureCanvasTkAgg(fig7, master=tab7).get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

    finestra.mainloop()


if __name__ == "__main__":
    dati_benchmark = esegui_benchmark()

    # <-- INIEZIONE DELL'ANALYZER
    dati_benchmark = analyzer(dati_benchmark)

    mostra_interfaccia_completa(dati_benchmark)