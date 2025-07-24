import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (import obbligatorio per 3-d)
from skimage import color, morphology, measure

# ---------------------- Config pagina -----------------------------------
st.set_page_config(page_title="Isolamento Bande Rosse", layout="wide")
st.title("Isola le bande rosse e analizza l’intensità")

# ---------------------- Upload -----------------------------------------
upl = st.file_uploader("Carica un’immagine", ["jpg", "jpeg", "png"])
if upl is None:
    st.info("Carica un’immagine per iniziare.")
    st.stop()

img_rgb  = Image.open(upl).convert("RGB")
arr_rgb  = np.asarray(img_rgb) / 255.0  # 0-1
h, w, _  = arr_rgb.shape
arr_red8 = (arr_rgb[:, :, 0] * 255).astype(np.uint8)

# ---------------------- Parametri soglia (sidebar) ----------------------
st.sidebar.header("Parametri HSV")
sat_thr  = st.sidebar.slider("Soglia saturazione", 0.0, 1.0, 0.4, 0.01)
val_thr  = st.sidebar.slider("Soglia luminosità", 0.0, 1.0, 0.2, 0.01)
min_area = st.sidebar.number_input("Area minima banda (px²)", 1, 10_000, 150)

# lobi di hue per il rosso
h_low1, h_high1 = 0/360, 15/360
h_low2, h_high2 = 345/360, 1

# ---------------------- Maschera HSV & morfologia -----------------------
hsv = color.rgb2hsv(arr_rgb)
H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

mask = (
    (((H >= h_low1) & (H <= h_high1)) | ((H >= h_low2) & (H <= h_high2)))
    & (S >= sat_thr) & (V >= val_thr)
)
mask = morphology.binary_closing(mask, morphology.disk(1))
mask = morphology.binary_opening(mask,  morphology.disk(1))

# ---------------------- Labeling & statistiche --------------------------
labels = measure.label(mask)
props  = measure.regionprops(labels, intensity_image=arr_rgb[:, :, 0])

rows  = []
overlay = img_rgb.copy()
draw   = ImageDraw.Draw(overlay)

for p in props:
    if p.area < min_area:
        continue
    minr, minc, maxr, maxc = p.bbox
    rows.append({
        "id":        p.label,
        "x_min":     minc,
        "x_max":     maxc - 1,
        "y_min":     (h - 1) - (maxr - 1),
        "y_max":     (h - 1) - minr,
        "area_px":   p.area,
        "R_medio":   round(p.mean_intensity * 255, 1),
    })
    draw.rectangle([(minc, minr), (maxc - 1, maxr - 1)],
                   outline=(0, 255, 0), width=1)

df_bande = pd.DataFrame(rows)

# ---------------------- Output tabella / CSV ----------------------------
st.subheader("Bande rosse rilevate")
if df_bande.empty:
    st.warning("Nessuna banda significativa con le soglie correnti.")
else:
    st.dataframe(df_bande, use_container_width=True)
    st.download_button("📥 Scarica CSV bande",
                       df_bande.to_csv(index=False).encode(),
                       "bande_rosse.csv", "text/csv")

# ---------------------- Overlay & maschera ------------------------------
col1, col2 = st.columns(2)
with col1:
    st.image(overlay, caption="Overlay (contorno verde)", use_column_width=True)
with col2:
    fig_mask, axm = plt.subplots()
    axm.imshow(mask[::-1], cmap="gray", extent=[0, w, 0, h],
               origin="lower", aspect="auto")
    axm.axis("off")
    axm.set_title("Maschera bande")
    st.pyplot(fig_mask)

# ---------------------- Grafico 3-D (pixel rossi) -----------------------
with st.expander("Grafico 3-D intensità rosso nei pixel della maschera"):
    ys, xs = np.nonzero(mask)           # coordinate immagine (orig alto-sx)
    rs     = arr_red8[ys, xs]

    if len(xs) == 0:
        st.info("Maschera vuota: nessun pixel rosso da plottare.")
    else:
        # origine in basso: invertiamo y
        ys_bottom = (h - 1) - ys

        max_show = 30_000
        if len(xs) > max_show:
            sel = np.random.choice(len(xs), max_show, replace=False)
            xs_s, ys_s, rs_s = xs[sel], ys_bottom[sel], rs[sel]
            st.write(f"Mostrati {max_show} pixel su {len(xs)} (campionati).")
        else:
            xs_s, ys_s, rs_s = xs, ys_bottom, rs

        fig3d = plt.figure(figsize=(6, 6))
        ax3d  = fig3d.add_subplot(111, projection='3d')
        ax3d.scatter(xs_s, ys_s, rs_s, c=rs_s, cmap="Reds", s=3)
        ax3d.set_xlabel("x [px]")
        ax3d.set_ylabel("y [px] (origine in basso)")
        ax3d.set_zlabel("Valore R")
        ax3d.set_title("Pixel delle bande rosse – vista 3-D")
        st.pyplot(fig3d)

