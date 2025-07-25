import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from skimage import color, morphology, measure
from sklearn.cluster import KMeans

# ------------------------------------------------- Config pagina -----
st.set_page_config(page_title="Bande rosse con palette", layout="wide")
st.title("Isolamento bande rosse usando la palette di riferimento")

# -------------------------- Upload immagini --------------------------
col_up1, col_up2 = st.columns(2)
with col_up1:
    up_img = st.file_uploader("① Immagine da analizzare", ["png", "jpg", "jpeg"])
with col_up2:
    up_pal = st.file_uploader("② Palette di rifermento (strip colori)", ["png", "jpg", "jpeg"])

if not (up_img and up_pal):
    st.info("Carica entrambe le immagini per procedere.")
    st.stop()

img_rgb = Image.open(up_img).convert("RGB")
arr_rgb = np.asarray(img_rgb) / 255.0
h, w, _ = arr_rgb.shape

# --------------------- 1. estrazione colori palette ------------------
pal_rgb = Image.open(up_pal).convert("RGB")
pal_arr = np.asarray(pal_rgb) / 255.0
pal_lab = color.rgb2lab(pal_arr)

# scarta pixel quasi bianchi (L>95, a≈b≈0)
mask_pal = pal_lab[:, :, 0] < 95
samples  = pal_lab[mask_pal].reshape(-1, 3)

n_colors = st.sidebar.slider("Quanti colori estrarre dalla palette?", 1, 10, 8)
kmeans   = KMeans(n_clusters=n_colors, n_init="auto", random_state=0)
centers  = kmeans.fit(samples).cluster_centers_    # colori LAB (forma n_colors,3)

# ordina dal più scuro al più chiaro (L crescente)
centers = centers[np.argsort(centers[:, 0])]

# --------------------- 2. ΔE* mappa immagine -------------------------
lab_img  = color.rgb2lab(arr_rgb)
delta_maps = [color.deltaE_cie2000(lab_img, c) for c in centers]
delta_min  = np.min(delta_maps, axis=0)

delta_thr = st.sidebar.slider("Tolleranza ΔE* max", 0.0, 50.0, 18.0, 0.1)
mask = delta_min < delta_thr

# opzionale: morfologia per pulire piccoli puntini
mask = morphology.remove_small_objects(mask, min_size=50)

# --------------------- 3. Label & statistiche bande ------------------
labels = measure.label(mask)
props  = measure.regionprops(labels, intensity_image=arr_rgb[:, :, 0])

rows, overlay = [], img_rgb.copy()
draw = ImageDraw.Draw(overlay)

for p in props:
    minr, minc, maxr, maxc = p.bbox
    rows.append({
        "id": p.label,
        "x_min": minc, "x_max": maxc - 1,
        "y_min": (h - 1) - (maxr - 1),
        "y_max": (h - 1) - minr,
        "area_px": p.area,
        "R_medio": round(p.mean_intensity * 255, 1)
    })
    draw.rectangle([(minc, minr), (maxc - 1, maxr - 1)],
                   outline=(0, 255, 0), width=1)

df_bande = pd.DataFrame(rows)

# --------------------- 4. Output tabella / CSV ------------------------
st.subheader("Bande rosse trovate")
if df_bande.empty:
    st.warning("Nessuna banda supera la soglia ΔE*.")
else:
    st.dataframe(df_bande, use_container_width=True)
    st.download_button("📥 Scarica CSV bande",
                       df_bande.to_csv(index=False).encode(),
                       "bande_rosse.csv", "text/csv")

# --------------------- 5. Overlay & maschera --------------------------
col_viz1, col_viz2 = st.columns(2)
with col_viz1:
    st.image(overlay, caption="Overlay contorni bande", use_column_width=True)
with col_viz2:
    fig_mask, axm = plt.subplots()
    axm.imshow(mask[::-1], cmap="gray", extent=[0, w, 0, h],
               origin="lower", aspect="auto")
    axm.set_title("Maschera bande")
    axm.axis("off")
    st.pyplot(fig_mask)

# --------------------- 6. Grafico 3-D intensità vs x,y ----------------
with st.expander("Grafico 3-D intensità R (solo pixel in maschera)"):
    ys_img, xs = np.nonzero(mask)
    if len(xs) == 0:
        st.info("Maschera vuota: niente da plottare.")
    else:
        rs = (arr_rgb[:, :, 0] * 255).astype(np.uint8)[ys_img, xs]
        ys = (h - 1) - ys_img

        max_show = 30_000
        if len(xs) > max_show:
            idx = np.random.choice(len(xs), max_show, replace=False)
            xs, ys, rs = xs[idx], ys[idx], rs[idx]
            st.write(f"Mostrati {max_show} punti su {len(mask.nonzero()[0])}.")

        fig3d = plt.figure(figsize=(6, 6))
        ax3d  = fig3d.add_subplot(111, projection='3d')
        ax3d.scatter(xs, ys, rs, c=rs, cmap="Reds", s=3)
        ax3d.set_xlabel("x [px]")
        ax3d.set_ylabel("y [px] (origine in basso)")
        ax3d.set_zlabel("Valore R")
        st.pyplot(fig3d)
