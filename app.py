import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from skimage import color, morphology, measure
from skimage.color import deltaE_ciede2000   # ← funzione corretta
from sklearn.cluster import KMeans

# ------------------------------------------------- Config pagina -----
st.set_page_config(page_title="Isolamento Bande Rosse", layout="wide")
st.title("Isola le bande rosse e analizza l’intensità")

# -------------------------- Upload immagini --------------------------
c1, c2 = st.columns(2)
with c1:
    up_img = st.file_uploader("① Immagine da analizzare", ["png", "jpg", "jpeg"])
with c2:
    up_pal = st.file_uploader("② Palette di riferimento", ["png", "jpg", "jpeg"])

if not (up_img and up_pal):
    st.info("Carica entrambe le immagini.")
    st.stop()

img_rgb = Image.open(up_img).convert("RGB")
arr_rgb = np.asarray(img_rgb) / 255.0
h, w, _ = arr_rgb.shape

# ---------- 1. estrai colori dalla palette ---------------------------
pal_rgb = Image.open(up_pal).convert("RGB")
pal_arr = np.asarray(pal_rgb) / 255.0
pal_lab = color.rgb2lab(pal_arr)

mask_pal = pal_lab[:, :, 0] < 95            # elimina quasi-bianchi
samples  = pal_lab[mask_pal].reshape(-1, 3)

n_colors = st.sidebar.slider("Colori da estrarre", 1, 10, 8)
centers  = KMeans(n_clusters=n_colors, n_init="auto", random_state=0).fit(samples).cluster_centers_
centers  = centers[np.argsort(centers[:, 0])]   # ordina per luminosità

# ---------- 2. ΔE* e maschera ---------------------------------------
lab_img   = color.rgb2lab(arr_rgb)
delta_stack = np.stack(
    [deltaE_ciede2000(lab_img, c.reshape(1, 1, 3))   # <- reshape (1,1,3)
     for c in centers],
    axis=0                                            # shape (n_colors, H, W)
)
delta_min = delta_stack.min(axis=0)                   # shape (H, W)
delta_thr = st.sidebar.slider("Tolleranza ΔE*", 0.0, 50.0, 18.0, 0.1)
mask = delta_min < delta_thr
mask = morphology.remove_small_objects(mask, min_size=50)

# ---------- 3. label & statistiche ----------------------------------
labels = measure.label(mask)
props  = measure.regionprops(labels, intensity_image=arr_rgb[:, :, 0])

rows, overlay = [], img_rgb.copy()
draw = ImageDraw.Draw(overlay)
for p in props:
    minr, minc, maxr, maxc = p.bbox
    rows.append({
        "id": p.label,
        "x_min": minc, "x_max": maxc - 1,
        "y_min": (h - 1) - (maxr - 1), "y_max": (h - 1) - minr,
        "area_px": p.area,
        "R_medio": round(p.mean_intensity * 255, 1),
    })
    draw.rectangle([(minc, minr), (maxc - 1, maxr - 1)],
                   outline=(0, 255, 0), width=1)

df_bande = pd.DataFrame(rows)

st.subheader("Bande rosse trovate")
if df_bande.empty:
    st.warning("Nessuna banda con la soglia corrente.")
else:
    st.dataframe(df_bande, use_container_width=True)
    st.download_button("📥 Scarica CSV", df_bande.to_csv(index=False).encode(),
                       "bande_rosse.csv", "text/csv")

# ---------- 4. overlay & maschera -----------------------------------
c3, c4 = st.columns(2)
with c3:
    st.image(overlay, caption="Overlay contorni", use_column_width=True)
with c4:
    fig_m, axm = plt.subplots()
    axm.imshow(mask[::-1], cmap="gray", extent=[0, w, 0, h],
               origin="lower", aspect="auto")
    axm.axis("off")
    axm.set_title("Maschera bande")
    st.pyplot(fig_m)

# ---------- 5. Grafico 3-D ------------------------------------------
with st.expander("Grafico 3-D intensità R (pixel maschera)"):
    ys_img, xs = np.nonzero(mask)
    if len(xs) == 0:
        st.info("Maschera vuota.")
    else:
        rs = (arr_rgb[:, :, 0] * 255).astype(np.uint8)[ys_img, xs]
        ys = (h - 1) - ys_img
        max_show = 30_000
        if len(xs) > max_show:
            idx = np.random.choice(len(xs), max_show, replace=False)
            xs, ys, rs = xs[idx], ys[idx], rs[idx]
            st.write(f"Campionati {max_show} pixel su {len(mask.nonzero()[0])}.")
        fig3d = plt.figure(figsize=(6, 6))
        ax3d = fig3d.add_subplot(111, projection='3d')
        ax3d.scatter(xs, ys, rs, c=rs, cmap="Reds", s=3)
        ax3d.set_xlabel("x [px]")
        ax3d.set_ylabel("y [px] (origine in basso)")
        ax3d.set_zlabel("Valore R")
        ax3d.set_title("Pixel bande rosse – 3-D")
        st.pyplot(fig3d)
