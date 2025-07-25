import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from skimage import color, morphology, measure
from skimage.color import deltaE_ciede2000
from sklearn.cluster import KMeans

# --------------- Config pagina ---------------------------------------
st.set_page_config(page_title="Bande rosse con palette", layout="wide")
st.title("Isolamento bande rosse basato su palette")

# --------------- Upload immagini -------------------------------------
c_up1, c_up2 = st.columns(2)
with c_up1:
    up_img = st.file_uploader("① Immagine da analizzare", ["png", "jpg", "jpeg"])
with c_up2:
    up_pal = st.file_uploader("② Palette di riferimento", ["png", "jpg", "jpeg"])

if not (up_img and up_pal):
    st.info("Carica entrambe le immagini.")
    st.stop()

img_rgb = Image.open(up_img).convert("RGB")
arr_rgb = np.asarray(img_rgb) / 255.0
h, w, _ = arr_rgb.shape

# --------------- Estrai colori dalla palette -------------------------
pal_rgb = Image.open(up_pal).convert("RGB")
pal_arr = np.asarray(pal_rgb) / 255.0
pal_lab = color.rgb2lab(pal_arr)
mask_pal = pal_lab[:, :, 0] < 95
samples = pal_lab[mask_pal].reshape(-1, 3)

st.sidebar.header("Parametri estrazione colori")
n_colors_sl = st.sidebar.slider("Colori da estrarre (slider)", 1, 10, 8)
n_colors_in = st.sidebar.number_input("Colori da estrarre (num.)", 1, 10, n_colors_sl)
n_colors = int(n_colors_in)

centers = KMeans(n_clusters=n_colors, n_init="auto", random_state=0).fit(samples).cluster_centers_
centers = centers[np.argsort(centers[:, 0])]  # ordina per luminosità

# --------------- ΔE* mappa -------------------------------------------
lab_img = color.rgb2lab(arr_rgb)
delta_stack = np.stack(
    [deltaE_ciede2000(lab_img, c.reshape(1, 1, 3)) for c in centers],
    axis=0
)
delta_min = delta_stack.min(axis=0)

# --------------- Parametri soglia ΔE* e morfologia -------------------
st.sidebar.header("Parametri filtro ΔE* & morfologia")
delta_thr_sl = st.sidebar.slider("Tolleranza ΔE* (slider)", 0.0, 50.0, 18.0, 0.1)
delta_thr_in = st.sidebar.number_input("Tolleranza ΔE* (num.)", 0.0, 50.0, delta_thr_sl, 0.1)
delta_thr = float(delta_thr_in)

min_area_sl = st.sidebar.slider("Area minima (slider)", 1, 10_000, 150)
min_area_in = st.sidebar.number_input("Area minima (num.)", 1, 10_000, min_area_sl)
min_area = int(min_area_in)

mask = delta_min < delta_thr
mask = morphology.remove_small_objects(mask, min_size=min_area)

# --------------- Label & statistiche bande ---------------------------
labels = measure.label(mask)
props = measure.regionprops(labels, intensity_image=arr_rgb[:, :, 0])

rows = []
overlay = img_rgb.copy()
draw = ImageDraw.Draw(overlay)

for p in props:
    minr, minc, maxr, maxc = p.bbox
    y_cent_img, x_cent = p.centroid         # centroid in coord immagine
    y_cent = (h - 1) - y_cent_img           # origine in basso
    rows.append({
        "id": p.label,
        "x_min": minc, "x_max": maxc - 1,
        "y_min": (h - 1) - (maxr - 1), "y_max": (h - 1) - minr,
        "x_c": round(x_cent, 1), "y_c": round(y_cent, 1),
        "area_px": p.area,
        "R_medio": round(p.mean_intensity * 255, 1),
    })
    draw.rectangle([(minc, minr), (maxc - 1, maxr - 1)],
                   outline=(0, 255, 0), width=1)

df_bande = pd.DataFrame(rows)

# --------------- Tabella & CSV ---------------------------------------
st.subheader("Bande rosse trovate")
if df_bande.empty:
    st.warning("Nessuna banda supera i criteri.")
else:
    st.dataframe(df_bande, use_container_width=True)
    st.download_button("📥 Scarica CSV bande",
                       df_bande.to_csv(index=False).encode(),
                       "bande_rosse.csv", "text/csv")

# --------------- Overlay & maschera ----------------------------------
col_v1, col_v2 = st.columns(2)
with col_v1:
    st.image(overlay, caption="Overlay contorni", use_column_width=True)
with col_v2:
    fig_m, axm = plt.subplots()
    axm.imshow(mask[::-1], cmap="gray", extent=[0, w, 0, h],
               origin="lower", aspect="auto")
    axm.axis("off")
    axm.set_title("Maschera bande")
    st.pyplot(fig_m)

# --------------- Grafico 2-D centri vs R_medio -----------------------
if not df_bande.empty:
    st.subheader("R medio in funzione dei centroidi bande")
    fig2, ax2 = plt.subplots()
    sc = ax2.scatter(df_bande["x_c"], df_bande["y_c"],
                     c=df_bande["R_medio"], cmap="Reds", s=80, edgecolors="k")
    ax2.set_xlabel("x centro banda [px]")
    ax2.set_ylabel("y centro banda [px] (origine in basso)")
    ax2.set_title("Centroidi bande – colore = R medio")
    cb = fig2.colorbar(sc, ax=ax2, label="R medio")
    ax2.invert_yaxis()            # y con origine in basso
    st.pyplot(fig2)
