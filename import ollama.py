import requests
import math
from datasets import load_dataset
import time
import os
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from matplotlib.ticker import MultipleLocator

os.environ["HF_TOKEN"] = "ProgettoSemestre"
start = time.time()
print("Scarico o carico il dataset dalla cache...")
dataset = load_dataset("google/boolq")

dati_validazione = dataset['validation'].shuffle()

NUM_TEST = 100
TEMPERATURA = 1.3
RIPETIZIONI_PER_DOMANDA = 10

TP = 0;
TN = 0;
FP = 0;
FN = 0

incertezza_corrette = []
incertezza_errate = []
entropia_corrette = []
entropia_errate = []

risultati_per_tabella = []

# Dizionari per tracciare le distribuzioni e le risposte medie
distribuzioni_totali = {}
distribuzioni_conteggio_domande = {}
distribuzioni_risposte_corrette = {}
distribuzioni_risposte_errate = {}


def entropia_binaria(p_true: float, p_false: float) -> float:
    """Shannon entropy (bit) di una distribuzione binaria normalizzata."""
    h = 0.0
    for p in (p_true, p_false):
        if p > 0:
            h -= p * math.log2(p)
    return h


print(f"\nInizio test su {NUM_TEST} domande (con {RIPETIZIONI_PER_DOMANDA} ripetizioni l'una)...")

url_ollama = 'http://localhost:11434/api/generate'

for i in range(NUM_TEST):
    riga = dati_validazione[i]
    testo = riga['passage']
    domanda = riga['question']
    risposta_reale = str(riga['answer']).lower()
    prompt_completo = (
        f"Basandoti sul seguente testo, rispondi alla domanda finale solo con 'True' o 'False'.\n\n"
        f"Testo: {testo}\n\n"
        f"Domanda: {domanda}\n"
        f"Risposta:"
    )

    payload = {
        'model': 'llama3',
        'prompt': prompt_completo,
        'stream': False,
        'options': {'num_predict': 10, 'temperature': TEMPERATURA},
        'raw': False,
        'logprobs': True,
        'top_logprobs': 5
    }

    print(f"\nElaborazione Domanda {i + 1}/{NUM_TEST}")

    conteggi_ripetizioni = {"true": 0, "false": 0, "errore": 0}
    resp_json_principale = {}
    risposta_ufficiale = "errore_modello"
    primo_successo = False

    running_true = 0
    running_false = 0
    entropie_cumulative = []

    # ── Accumulatori per la media delle probabilità ──
    somma_prob_true = 0.0
    somma_prob_false = 0.0
    conteggio_prob_valide = 0

    # ── Vettore delle distribuzioni (T, F) ──
    distribuzioni_ripetizioni = []

    for rep in range(RIPETIZIONI_PER_DOMANDA):
        try:
            response = requests.post(url_ollama, json=payload)
            response.raise_for_status()
            resp_json = response.json()
        except Exception as e:
            print(f"Errore di connessione a Ollama (Ripetizione {rep + 1}): {e}")
            continue

        testo_generato = resp_json.get('response', '').strip().lower()
        risposta_pulita = testo_generato.replace(".", "")

        if not primo_successo:
            resp_json_principale = resp_json
            if "true" in risposta_pulita:
                risposta_ufficiale = "true"
            elif "false" in risposta_pulita:
                risposta_ufficiale = "false"
            else:
                risposta_ufficiale = "errore_modello"
            primo_successo = True

        if "true" in risposta_pulita:
            conteggi_ripetizioni["true"] += 1
        elif "false" in risposta_pulita:
            conteggi_ripetizioni["false"] += 1
        else:
            conteggi_ripetizioni["errore"] += 1

        print(f"  Ripetizione {rep + 1}: {risposta_pulita}")

        if "true" in risposta_pulita:
            running_true += 1
        elif "false" in risposta_pulita:
            running_false += 1

        k_valide = running_true + running_false
        if k_valide > 0:
            p_t = running_true / k_valide
            p_f = running_false / k_valide
            entropie_cumulative.append(entropia_binaria(p_t, p_f))

        # ── Estrazione logprobs ──
        p_t_rep = 0.0
        p_f_rep = 0.0
        if 'logprobs' in resp_json and isinstance(resp_json.get('logprobs'), list) and len(resp_json['logprobs']) > 0:
            primo_token_data = resp_json['logprobs'][0] if isinstance(resp_json['logprobs'], list) else {}
            top_probs_dict = primo_token_data.get('top_logprobs', [])
            for token_dict in top_probs_dict:
                token_text = token_dict.get('token', '').strip().lower()
                logp = token_dict.get('logprob', -100)
                if "true" in token_text:
                    p_t_rep += math.exp(logp)
                elif "false" in token_text:
                    p_f_rep += math.exp(logp)

            somma_rep = p_t_rep + p_f_rep
            if somma_rep > 0:
                p_t_norm = p_t_rep / somma_rep
                p_f_norm = p_f_rep / somma_rep

                somma_prob_true += p_t_norm
                somma_prob_false += p_f_norm
                conteggio_prob_valide += 1

                distribuzioni_ripetizioni.append(f"[T:{p_t_norm * 100:.1f}% F:{p_f_norm * 100:.1f}%]")
            else:
                distribuzioni_ripetizioni.append("[0.50, 0.50]")

    vettore_str = ", ".join(distribuzioni_ripetizioni)
    print(f"  -> Vettore distribuzioni (T, F): [{vettore_str}]")

    # Calcoli entropia e distribuzioni finali
    if entropie_cumulative:
        h_min = min(entropie_cumulative)
        h_max = max(entropie_cumulative)
    else:
        h_min = h_max = 0.0

    tot_valide = running_true + running_false
    if tot_valide > 0:
        dist_media_t = (running_true / tot_valide) * 100
        dist_media_f = (running_false / tot_valide) * 100
    else:
        dist_media_t = dist_media_f = 50.0

    h_media_val = entropia_binaria(dist_media_t / 100, dist_media_f / 100)

    valori = [conteggi_ripetizioni["true"], conteggi_ripetizioni["false"]]
    valori.sort(reverse=True)
    chiave_distribuzione = f"{valori[0]}-{valori[1]}"
    distribuzioni_totali[chiave_distribuzione] = distribuzioni_totali.get(chiave_distribuzione, 0) + 1

    # ── Probabilità 1° Ripetizione (per confronto) ──
    prob_true_raw = 0.0
    prob_false_raw = 0.0
    if 'logprobs' in resp_json_principale and isinstance(resp_json_principale.get('logprobs'), list) and len(
            resp_json_principale['logprobs']) > 0:
        primo_token_data = resp_json_principale['logprobs'][0] if isinstance(resp_json_principale['logprobs'],
                                                                             list) else {}
        top_probs_dict = primo_token_data.get('top_logprobs', [])
        for token_dict in top_probs_dict:
            token_text = token_dict.get('token', '').strip().lower()
            logp = token_dict.get('logprob', -100)
            if "true" in token_text:
                prob_true_raw += math.exp(logp)
            elif "false" in token_text:
                prob_false_raw += math.exp(logp)

    somma_raw = prob_true_raw + prob_false_raw
    if somma_raw > 0:
        perc_true_1 = (prob_true_raw / somma_raw) * 100
        perc_false_1 = (prob_false_raw / somma_raw) * 100
    else:
        perc_true_1 = perc_false_1 = 50.0

    # ── Calcolo Media delle Probabilità ──
    if conteggio_prob_valide > 0:
        perc_true_avg = (somma_prob_true / conteggio_prob_valide) * 100
        perc_false_avg = (somma_prob_false / conteggio_prob_valide) * 100
    else:
        perc_true_avg = perc_false_avg = 50.0

    sicurezza = max(perc_true_avg, perc_false_avg)
    incertezza_perc = 100.0 - sicurezza

    # ── SALVATAGGIO ESITI ──
    esito_corretto = (risposta_ufficiale == risposta_reale)
    if esito_corretto:
        incertezza_corrette.append(incertezza_perc)
        entropia_corrette.append(h_media_val)
        if risposta_reale == "true":
            TP += 1
        else:
            TN += 1
    else:
        incertezza_errate.append(incertezza_perc)
        entropia_errate.append(h_media_val)
        if risposta_reale == "false":
            FP += 1
        else:
            FN += 1

    # ── SALVATAGGIO RISPOSTE PER DISTRIBUZIONE ──
    if risposta_reale == "true":
        risposte_giuste = conteggi_ripetizioni["true"]
        risposte_sbagliate = conteggi_ripetizioni["false"]
    elif risposta_reale == "false":
        risposte_giuste = conteggi_ripetizioni["false"]
        risposte_sbagliate = conteggi_ripetizioni["true"]
    else:
        risposte_giuste = 0
        risposte_sbagliate = 0

    distribuzioni_conteggio_domande[chiave_distribuzione] = distribuzioni_conteggio_domande.get(chiave_distribuzione,
                                                                                                0) + 1
    distribuzioni_risposte_corrette[chiave_distribuzione] = distribuzioni_risposte_corrette.get(chiave_distribuzione,
                                                                                                0) + risposte_giuste
    distribuzioni_risposte_errate[chiave_distribuzione] = distribuzioni_risposte_errate.get(chiave_distribuzione,
                                                                                            0) + risposte_sbagliate

    inizio_domanda = " ".join(domanda.split()[:4]) + "..."
    testo_probabilita_avg = f"T:{perc_true_avg:.1f}% F:{perc_false_avg:.1f}%"
    testo_distribuzione = chiave_distribuzione

    risultati_per_tabella.append({
        "domanda": inizio_domanda,
        "prob_media": testo_probabilita_avg,
        "generata_dist": testo_distribuzione,
        "reale": risposta_reale,
        "corretta": esito_corretto,
        "h_min": h_min,
        "h_max": h_max,
        "h_media": h_media_val,
    })

end = time.time()
print(f"\nTest completato in {end - start:.1f} secondi! Genero l'interfaccia...")


# ============================================================
# INTERFACCIA
# ============================================================
def mostra_interfaccia_completa():
    finestra = tk.Tk()
    finestra.title(f"Report Benchmark ({RIPETIZIONI_PER_DOMANDA} Ripetizioni su {NUM_TEST} Domande)")
    finestra.geometry("1100x650")

    notebook = ttk.Notebook(finestra)
    notebook.pack(fill='both', expand=True, padx=10, pady=10)

    # ==========================================
    # SCHEDA 1: Tabella e Matrice
    # ==========================================
    tab1 = ttk.Frame(notebook)
    notebook.add(tab1, text="Dati e Matrice")

    totale_corrette = TP + TN
    percentuale_corrette = (totale_corrette / NUM_TEST) * 100
    tk.Label(tab1,
             text=f"Risultati: {totale_corrette}/{NUM_TEST} Corrette ({percentuale_corrette:.1f}%)",
             font=("Helvetica", 14, "bold")).pack(pady=10)

    frame_matrice = tk.Frame(tab1)
    frame_matrice.pack(side=tk.BOTTOM, pady=10)
    tk.Label(frame_matrice, text="Matrice di Confusione", font=("Helvetica", 12, "bold")).grid(row=0, column=0,
                                                                                               columnspan=3, pady=5)
    tk.Label(frame_matrice, text="Modello: True").grid(row=1, column=1)
    tk.Label(frame_matrice, text="Modello: False").grid(row=1, column=2)
    tk.Label(frame_matrice, text="Realtà: True").grid(row=2, column=0)
    tk.Label(frame_matrice, text=f"TP\n{TP}", bg="#c6efce", width=10, height=2, relief="groove").grid(row=2, column=1)
    tk.Label(frame_matrice, text=f"FN\n{FN}", bg="#ffc7ce", width=10, height=2, relief="groove").grid(row=2, column=2)
    tk.Label(frame_matrice, text="Realtà: False").grid(row=3, column=0)
    tk.Label(frame_matrice, text=f"FP\n{FP}", bg="#ffc7ce", width=10, height=2, relief="groove").grid(row=3, column=1)
    tk.Label(frame_matrice, text=f"TN\n{TN}", bg="#c6efce", width=10, height=2, relief="groove").grid(row=3, column=2)

    frame_tabella = tk.Frame(tab1)
    frame_tabella.pack(side=tk.TOP, fill="both", expand=True, padx=10, pady=5)

    colonne = ("Domanda", "ProbMedia", "Distribuzione", "Reale",
               "H_min", "H_max", "H_media")
    tabella = ttk.Treeview(frame_tabella, columns=colonne, show="headings")

    tabella.heading("Domanda", text="Inizio Domanda");
    tabella.column("Domanda", width=160)
    tabella.heading("ProbMedia", text=f"Media Prob. [{RIPETIZIONI_PER_DOMANDA} Rip]");
    tabella.column("ProbMedia", width=135, anchor="center")
    tabella.heading("Distribuzione", text="Dist. [Rip.]");
    tabella.column("Distribuzione", width=70, anchor="center")
    tabella.heading("Reale", text="Reale");
    tabella.column("Reale", width=50, anchor="center")
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

    for riga in risultati_per_tabella:
        colore = "verde" if riga["corretta"] else "rosso"
        tabella.insert("", tk.END,
                       values=(
                           riga["domanda"],
                           riga["prob_media"],
                           riga["generata_dist"],
                           riga["reale"],
                           f"{riga['h_min']:.3f}",
                           f"{riga['h_max']:.3f}",
                           f"{riga['h_media']:.2f}",
                       ),
                       tags=(colore,))

    # ==========================================
    # SCHEDA 2: Grafico Box Plot + Strip Plot — ENTROPIA
    # ==========================================
    tab2 = ttk.Frame(notebook)
    notebook.add(tab2, text="Grafico Entropia (Box + Strip)")

    fig = Figure(figsize=(7, 4), dpi=100)
    ax = fig.add_subplot(111)

    dati_plot = [];
    etichette = []

    for val in entropia_corrette:
        dati_plot.append(val);
        etichette.append("Corrette")
    for val in entropia_errate:
        dati_plot.append(val);
        etichette.append("Errate")
    for val in entropia_corrette + entropia_errate:
        dati_plot.append(val);
        etichette.append("Aggregato")

    if dati_plot:
        # 1. Box plot in background (colori tenui). fliersize=0 nasconde i punti anomali di default
        sns.boxplot(
            x=etichette, y=dati_plot, hue=etichette,
            palette={"Corrette": "#c6efce", "Errate": "#ffc7ce", "Aggregato": "#e0e0e0"},
            order=["Corrette", "Errate", "Aggregato"],
            ax=ax, legend=False, fliersize=0
        )

        # 2. Strip plot in foreground (colori scuri, mostrano l'esatta posizione di tutti i punti)
        sns.stripplot(
            x=etichette, y=dati_plot, hue=etichette,
            palette={"Corrette": "#006100", "Errate": "#9c0006", "Aggregato": "#333333"},
            order=["Corrette", "Errate", "Aggregato"],
            ax=ax, legend=False, jitter=True, alpha=0.7, size=5
        )

    ax.set_title("Entropia Media del Modello (bit)")
    ax.set_ylabel("Entropia H (bit)  [0 = certo, 1 = massima incertezza]")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    canvas = FigureCanvasTkAgg(fig, master=tab2)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

    # ==========================================
    # PREPARAZIONE DATI PER SCHEDA 3 E 4 (Medie Generiche)
    # ==========================================
    categorie_x = []
    for k in range(RIPETIZIONI_PER_DOMANDA, math.ceil(RIPETIZIONI_PER_DOMANDA / 2) - 1, -1):
        categorie_x.append(f"{k}-{RIPETIZIONI_PER_DOMANDA - k}")

    for key in sorted(distribuzioni_conteggio_domande.keys(), reverse=True):
        if key not in categorie_x:
            categorie_x.append(key)

    medie_giuste = []
    medie_sbagliate = []

    for cat in categorie_x:
        num_domande = distribuzioni_conteggio_domande.get(cat, 0)
        if num_domande > 0:
            medie_giuste.append(distribuzioni_risposte_corrette.get(cat, 0) / num_domande)
            medie_sbagliate.append(distribuzioni_risposte_errate.get(cat, 0) / num_domande)
        else:
            medie_giuste.append(0)
            medie_sbagliate.append(0)

    # ==========================================
    # SCHEDA 3: Istogramma Colonna Unica (Media Risposte)
    # ==========================================
    tab3 = ttk.Frame(notebook)
    notebook.add(tab3, text="Media Risposte")

    fig3 = Figure(figsize=(6, 4), dpi=100)
    ax3 = fig3.add_subplot(111)

    ax3.bar(categorie_x, medie_giuste, color='#4CAF50', label='Risposte Giuste (Media)')
    ax3.bar(categorie_x, medie_sbagliate, bottom=medie_giuste, color='#F44336', label='Risposte Sbagliate (Media)')

    ax3.set_title(f"Media Risposte Giuste vs Sbagliate per Distribuzione (su {RIPETIZIONI_PER_DOMANDA} Ripetizioni)")
    ax3.set_xlabel("Combinazione (Maggioranza - Minoranza)")
    ax3.set_ylabel(f"Numero Medio di Risposte (Max {RIPETIZIONI_PER_DOMANDA})")
    ax3.legend()

    step_y = max(1, RIPETIZIONI_PER_DOMANDA // 10)
    ax3.yaxis.set_major_locator(MultipleLocator(step_y))
    ax3.grid(axis='y', which='major', linestyle='-', linewidth=0.8, alpha=0.7)

    canvas3 = FigureCanvasTkAgg(fig3, master=tab3)
    canvas3.draw()
    canvas3.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

    # ==========================================
    # SCHEDA 4: Istogramma Colonne Affiancate (Percentuali Risposte)
    # ==========================================
    tab4 = ttk.Frame(notebook)
    notebook.add(tab4, text="Percentuali Risposte")

    perc_giuste = []
    perc_sbagliate = []

    for g, s in zip(medie_giuste, medie_sbagliate):
        totale = g + s
        if totale > 0:
            perc_giuste.append((g / totale) * 100)
            perc_sbagliate.append((s / totale) * 100)
        else:
            perc_giuste.append(0)
            perc_sbagliate.append(0)

    fig4 = Figure(figsize=(6, 4), dpi=100)
    ax4 = fig4.add_subplot(111)

    x = range(len(categorie_x))
    width = 0.35

    x_giuste = [i - width / 2 for i in x]
    x_sbagliate = [i + width / 2 for i in x]

    ax4.bar(x_giuste, perc_giuste, width, color='#4CAF50', label='Risposte Giuste (%)')
    ax4.bar(x_sbagliate, perc_sbagliate, width, color='#F44336', label='Risposte Sbagliate (%)')

    ax4.set_xticks(x)
    ax4.set_xticklabels(categorie_x)
    ax4.set_title(f"Percentuale Media di Risposte Giuste/Sbagliate per Distribuzione")
    ax4.set_xlabel("Combinazione (Maggioranza - Minoranza)")
    ax4.set_ylabel("Percentuale (%)")
    ax4.set_ylim(0, 105)
    ax4.legend()
    ax4.yaxis.set_major_locator(MultipleLocator(10))
    ax4.grid(axis='y', linestyle='--', alpha=0.7)

    canvas4 = FigureCanvasTkAgg(fig4, master=tab4)
    canvas4.draw()
    canvas4.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

    finestra.mainloop()

mostra_interfaccia_completa()