# Actividad 9. Identificación estocástica de sistemas
# Método de mínimos cuadrados para estimar a, b, c y d.
# Modelo supuesto: y[n] = a*y[n-1] + b*y[n-2] + c*x[n-1] + d*x[n-2]
# Nota: la captura parece repetir x[n-1] en el último término. Se usa x[n-2]
# como estructura ARX de segundo orden; si la ecuación realmente repite x[n-1],
# c y d no se pueden identificar por separado porque sus columnas serían iguales.

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Actividad 9 | Identificación de sistemas", page_icon="📈", layout="wide")

st.title("📈 Actividad 9. Identificación estocástica de sistemas")
st.write(
    "Estimación de parámetros por mínimos cuadrados, comparación de la respuesta real "
    "con la estimada y análisis de residuos."
)

DEFAULT_DATA = """n,y,x
0,0.0000,1
1,2.0000,1
2,3.2000,1
3,3.4000,1
4,2.7920,1
5,1.7928,1
6,0.8438,1
7,0.2547,1
8,0.1321,1
9,0.3965,1
10,0.8573,1
11,1.3069,1
12,1.5952,1
"""

with st.sidebar:
    st.header("Configuración")
    st.caption("Puedes editar los datos de entrada-salida.")
    raw_data = st.text_area("Datos CSV (columnas n,y,x)", value=DEFAULT_DATA, height=300)
    model_choice = st.selectbox(
        "Estructura del modelo",
        ["Segundo orden: c·x[n−1] + d·x[n−2]", "Entrada repetida: (c+d)·x[n−1]"],
        help="La segunda opción reproduce literalmente la ecuación tal como aparece en la captura."
    )
    st.caption("Condiciones iniciales asumidas: y[-1] = 0, x[-1] = 0.")

try:
    data = pd.read_csv(io.StringIO(raw_data.strip()))
    required = {"n", "y", "x"}
    if not required.issubset(data.columns):
        raise ValueError("El CSV debe tener las columnas n,y,x.")
    data = data[["n", "y", "x"]].apply(pd.to_numeric, errors="raise").sort_values("n")
    if len(data) < 5:
        raise ValueError("Ingresa al menos 5 filas de datos.")
    if data[["n", "y", "x"]].isna().any().any():
        raise ValueError("No puede haber celdas vacías.")
except Exception as exc:
    st.error(f"No se pudieron leer los datos: {exc}")
    st.stop()

n = data["n"].to_numpy(dtype=int)
y = data["y"].to_numpy(dtype=float)
x = data["x"].to_numpy(dtype=float)
if not np.all(np.diff(n) == 1):
    st.error("La columna n debe contener muestras consecutivas, sin saltos.")
    st.stop()

# Acceso seguro a muestras anteriores. Para índices negativos se usa cero.
def past(arr, k):
    return float(arr[k]) if k >= 0 else 0.0

# Construcción de Phi y Y: y[n] = Phi[n] @ theta
rows, target, used_n = [], [], []
for i in range(1, len(data)):
    if model_choice.startswith("Segundo"):
        row = [past(y, i-1), past(y, i-2), past(x, i-1), past(x, i-2)]
    else:
        row = [past(y, i-1), past(y, i-2), past(x, i-1), past(x, i-1)]
    rows.append(row)
    target.append(y[i])
    used_n.append(n[i])

Phi = np.asarray(rows, dtype=float)
Y = np.asarray(target, dtype=float)
theta, residuals_lstsq, rank, singular_values = np.linalg.lstsq(Phi, Y, rcond=None)
a, b, c, d = theta
Y_one_step = Phi @ theta
residual_vector = Y - Y_one_step
sse = float(np.sum(residual_vector**2))
mse = float(np.mean(residual_vector**2))
rmse = float(np.sqrt(mse))
sst = float(np.sum((Y - np.mean(Y))**2))
r2 = 1.0 - sse/sst if sst > 0 else float("nan")

# Predicción recursiva (simulación libre): usa salidas previamente estimadas.
y_sim = np.zeros_like(y, dtype=float)
y_sim[0] = y[0]
for i in range(1, len(y)):
    y1 = y_sim[i-1]
    y2 = y_sim[i-2] if i-2 >= 0 else 0.0
    x1 = past(x, i-1)
    x2 = past(x, i-2)
    if model_choice.startswith("Segundo"):
        y_sim[i] = a*y1 + b*y2 + c*x1 + d*x2
    else:
        y_sim[i] = a*y1 + b*y2 + c*x1 + d*x1

st.subheader("1. Datos proporcionados")
st.dataframe(data, use_container_width=True, hide_index=True)

st.subheader("2. Matriz de datos y vector de salida")
left, right = st.columns(2)
with left:
    st.markdown("**Matriz Φ**")
    phi_df = pd.DataFrame(Phi, columns=["y[n−1]", "y[n−2]", "x[n−1]", "x[n−2]" if model_choice.startswith("Segundo") else "x[n−1]"])
    phi_df.insert(0, "n", used_n)
    st.dataframe(phi_df, use_container_width=True, hide_index=True)
with right:
    st.markdown("**Vector Y**")
    st.dataframe(pd.DataFrame({"n": used_n, "y[n]": Y}), use_container_width=True, hide_index=True)

st.latex(r"\\hat{\\theta} = (\\Phi^T\\Phi)^{-1}\\Phi^T Y \\quad\\text{(si existe la inversa)}")
st.write("En el programa se utiliza `numpy.linalg.lstsq`, que calcula la solución de mínimos cuadrados de forma numéricamente más robusta, incluso si la matriz no es invertible.")

if rank < Phi.shape[1]:
    st.warning(
        f"La matriz Φ tiene rango {rank} de {Phi.shape[1]}. No todos los parámetros son identificables "
        "por separado con estos datos/esta estructura. En particular, si se repite x[n−1], c y d quedan confundidos."
    )

st.subheader("3. Parámetros estimados")
cols = st.columns(4)
for col, name, value in zip(cols, ["a", "b", "c", "d"], theta):
    col.metric(name, f"{value:.6f}")
if model_choice.startswith("Segundo"):
    st.latex(
        rf"\\boxed{{y[n] = ({a:.5f})y[n-1] + ({b:.5f})y[n-2] + "
        rf"({c:.5f})x[n-1] + ({d:.5f})x[n-2]}}"
    )
else:
    st.latex(
        rf"\\boxed{{y[n] = ({a:.5f})y[n-1] + ({b:.5f})y[n-2] + "
        rf"({c+d:.5f})x[n-1]}}"
    )

st.subheader("4. Comparación: respuesta real vs. estimada")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(n, y, "o-", label="Respuesta real")
ax.plot(n, y_sim, "s--", label="Respuesta estimada (recursiva)")
ax.set_xlabel("Muestra n")
ax.set_ylabel("Salida y[n]")
ax.set_title("Identificación del sistema por mínimos cuadrados")
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.subheader("5. Validación del ajuste")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Suma de errores² (SSE)", f"{sse:.6f}")
m2.metric("RMSE (1 paso)", f"{rmse:.6f}")
m3.metric("R² (1 paso)", f"{r2:.4f}")
m4.metric("Rango de Φ", f"{rank}/{Phi.shape[1]}")

fig2, ax2 = plt.subplots(figsize=(10, 3.5))
ax2.axhline(0, linewidth=1)
ax2.stem(used_n, residual_vector, linefmt="C1-", markerfmt="C1o", basefmt="k-")
ax2.set_xlabel("Muestra n")
ax2.set_ylabel("Residuo e[n]")
ax2.set_title("Residuos de la predicción a un paso: y[n] − ŷ[n]")
ax2.grid(True, alpha=0.3)
st.pyplot(fig2, use_container_width=True)
plt.close(fig2)

resid_df = pd.DataFrame({
    "n": used_n,
    "y_real": Y,
    "y_estimado_1_paso": Y_one_step,
    "residuo": residual_vector
})
st.dataframe(resid_df, use_container_width=True, hide_index=True)
st.download_button(
    "⬇️ Descargar tabla de resultados CSV",
    data=resid_df.to_csv(index=False).encode("utf-8"),
    file_name="resultados_actividad9.csv",
    mime="text/csv"
)

st.subheader("6. Interpretación")
st.markdown(
    f"""
- Se construyó una matriz Φ con **{len(Y)} ecuaciones** y cuatro columnas de regresores.
- Los parámetros se calcularon minimizando la suma de los cuadrados de los residuos.
- El RMSE a un paso es **{rmse:.6f}**; cuanto menor sea, menor es el error de predicción en los datos usados.
- El valor de R² a un paso es **{r2:.4f}**. Se interpreta junto con los residuos y no como única prueba de validez.
- La gráfica recursiva es una prueba adicional: como utiliza salidas estimadas anteriores, sus errores pueden acumularse.
"""
)
st.info(
    "Importante para el reporte: la ecuación de la captura muestra dos veces x[n−1]. "
    "La estructura ARX usual de segundo orden utiliza x[n−1] y x[n−2]. "
    "Confirma con el profesor cuál estructura pide. Con una entrada escalón, separar efectos de entrada "
    "puede ser limitado; para identificar mejor parámetros de entrada se recomienda usar una señal de prueba "
    "más informativa."
)

st.caption("Actividad 9 · Identificación estocástica de sistemas · Python, NumPy, Pandas, Matplotlib y Streamlit")
