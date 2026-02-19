# ia_nacion_servicios.py
# IA – Nación Servicios (Liquidaciones)
# Herramienta para uso interno - AIE San Justo

import io
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# ---------------- UI / assets ----------------
HERE = Path(__file__).parent
ASSETS = HERE / "assets"
LOGO = ASSETS / "logo_aie.png"
FAVICON = ASSETS / "favicon-aie.ico"

st.set_page_config(
    page_title="IA – Nación Servicios",
    page_icon=str(FAVICON) if FAVICON.exists() else None,
    layout="centered",
)

if LOGO.exists():
    st.image(str(LOGO), width=220)

st.title("IA – Nación Servicios")

st.markdown(
    """
    <style>
      .block-container { max-width: 980px; padding-top: 2rem; padding-bottom: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- deps diferidas ----------------
try:
    import pdfplumber
except Exception as e:
    st.error(f"No se pudo importar pdfplumber: {e}\nRevisá requirements.txt")
    st.stop()

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False


# ---------------- utils ----------------
def money_to_float(raw: str):
    """
    Convierte montos como:
      $89,813.12  |  105.068,00  |  1.234,56  |  1234.56
    a float.
    """
    if raw is None:
        return np.nan
    s = str(raw).strip()
    if not s:
        return np.nan

    s = s.replace("$", "").replace(" ", "")

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]

    # Si tiene , y . decidimos decimal por el último separador
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            # 105.068,00 -> 105068.00
            s = s.replace(".", "").replace(",", ".")
        else:
            # 89,813.12 -> 89813.12
            s = s.replace(",", "")
    else:
        # solo coma
        if "," in s:
            parts = s.split(",")
            if len(parts[-1]) == 2 and all(p.isdigit() for p in parts):
                s = "".join(parts[:-1]) + "." + parts[-1]
            else:
                s = s.replace(",", "")

    try:
        v = float(s)
        return -v if neg else v
    except Exception:
        return np.nan


def fmt_ar(n) -> str:
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "—"
    return f"{n:,.2f}".replace(",", "§").replace(".", ",").replace("§", ".")


def _text_from_pdf(file_like) -> str:
    try:
        with pdfplumber.open(file_like) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages)
    except Exception:
        return ""


def _sum_matches(pattern: str, text: str) -> float:
    tot = 0.0
    for m in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
        v = money_to_float(m.group(1))
        if not np.isnan(v):
            tot += float(v)
    return float(tot)


def _first(pattern: str, text: str):
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def parse_liquidacion_page(text: str) -> dict:
    """
    Extrae una liquidación desde una página 'LIQUIDACION A COMERCIO':
    - TOTAL PRESENTADO / TOTAL DESCUENTO / SALDO
    - IVA 21%
    - Neto 21% (Gastos Comisiones) = Arancel + Arancel Costo Financiero + CARGO FINANCIERO
    - Retenciones IIBB = suma de líneas RET IIBB ... $<monto>
    """
    fecha_emision = _first(r"Fecha de Emisión\s+(\d{2}/\d{2}/\d{4})", text)
    fecha_pago = _first(r"Fecha de Pago:\s*(\d{2}/\d{2}/\d{4})", text)
    nro_liq = _first(r"Nro\. de Liquidación\s*([0-9]+)", text)

    establecimiento = _first(r"Establecimiento\s+(.+?)\s*(?:Domicilio Fiscal|CUIT Establecimiento|$)", text)
    cuit_est = _first(r"CUIT Establecimiento\s+([0-9\-]+)", text)

    total_presentado = money_to_float(_first(r"TOTAL PRESENTADO\s+\$?([0-9\.,]+)", text))
    total_descuento = money_to_float(_first(r"TOTAL DESCUENTO\s+\$?([0-9\.,]+)", text))
    saldo = money_to_float(_first(r"SALDO\s+\$?([0-9\.,]+)", text))

    # Detalle descuentos (en estas páginas los importes aparecen con $ antes del monto)
    arancel = _sum_matches(r"^\s*Arancel\s+\$\s*([0-9\.,]+)\s*$", text)
    arancel_cf = _sum_matches(r"^\s*Arancel Costo Financiero\s+\$\s*([0-9\.,]+)\s*$", text)
    cargo_fin = _sum_matches(r"^\s*CARGO FINANCIERO.*?\$\s*([0-9\.,]+)\s*$", text)
    iva_21 = _sum_matches(r"^\s*Iva 21%\s+\$\s*([0-9\.,]+)\s*$", text)

    # IMPORTANTE: en RET IIBB la línea trae porcentaje + $monto, capturamos el monto DESPUÉS del $
    ret_iibb = _sum_matches(r"^\s*RET\s*IIBB.*?\$\s*([0-9\.,]+)\s*$", text)

    neto_21 = arancel + arancel_cf + cargo_fin
    gasto_21_total = neto_21 + iva_21

    # Controles por liquidación
    descuentos_clasificados = neto_21 + iva_21 + ret_iibb
    otros_desc = (total_descuento - descuentos_clasificados) if not np.isnan(total_descuento) else np.nan
    ctrl_presentado = (total_presentado - total_descuento - saldo) if (not np.isnan(total_presentado) and not np.isnan(total_descuento) and not np.isnan(saldo)) else np.nan

    # check neto vs IVA/0.21 (solo informativo)
    neto_from_iva = (iva_21 / 0.21) if iva_21 else 0.0
    ctrl_neto = neto_21 - neto_from_iva if iva_21 else 0.0

    return {
        "Fecha Emisión": fecha_emision,
        "Fecha Pago": fecha_pago,
        "Nro Liquidación": nro_liq,
        "Establecimiento": establecimiento,
        "CUIT Establecimiento": cuit_est,

        "Total Presentado": float(total_presentado) if not np.isnan(total_presentado) else np.nan,
        "Total Descuento": float(total_descuento) if not np.isnan(total_descuento) else np.nan,
        "Saldo": float(saldo) if not np.isnan(saldo) else np.nan,

        "Gastos Comisiones Neto 21%": float(neto_21),
        "IVA 21%": float(iva_21),
        "Gasto 21% Total (Neto+IVA)": float(gasto_21_total),
        "Ret IIBB": float(ret_iibb),

        "Otros descuentos (no clasificados)": float(otros_desc) if not np.isnan(otros_desc) else np.nan,

        "Control: Presentado - Descuento - Saldo": float(ctrl_presentado) if not np.isnan(ctrl_presentado) else np.nan,
        "Control: Neto - IVA/0.21": float(ctrl_neto),
    }


@st.cache_data(show_spinner=False)
def process_pdf(data: bytes) -> pd.DataFrame:
    rows = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for p in pdf.pages:
            t = p.extract_text() or ""
            up = t.upper()
            if "LIQUIDACION A COMERCIO" in up and "NRO. DE LIQUIDACIÓN" in up:
                rows.append(parse_liquidacion_page(t))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["_fecha_pago"] = pd.to_datetime(df["Fecha Pago"], dayfirst=True, errors="coerce")
    df = df.sort_values(["_fecha_pago", "Nro Liquidación"]).drop(columns=["_fecha_pago"]).reset_index(drop=True)
    return df


def resumen_operativo(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Concepto", "Importe"])

    total_presentado = float(df["Total Presentado"].fillna(0).sum())
    total_descuento = float(df["Total Descuento"].fillna(0).sum())
    total_saldo = float(df["Saldo"].fillna(0).sum())

    neto_21 = float(df["Gastos Comisiones Neto 21%"].fillna(0).sum())
    iva_21 = float(df["IVA 21%"].fillna(0).sum())
    ret_iibb = float(df["Ret IIBB"].fillna(0).sum())

    clasificados = neto_21 + iva_21 + ret_iibb
    otros = total_descuento - clasificados

    ctrl_periodo = total_presentado - total_descuento - total_saldo

    out = [
        ["COMISIONES (NETO 21%)", neto_21],
        ["IVA 21%", iva_21],
        ["RETENCIONES IIBB", ret_iibb],
        ["OTROS DESCUENTOS (NO CLASIFICADOS)", otros],
        ["—", 0.0],
        ["TOTAL DESCUENTO (SEGÚN PDF)", total_descuento],
        ["TOTAL PRESENTADO (SEGÚN PDF)", total_presentado],
        ["SALDO (SEGÚN PDF)", total_saldo],
        ["CONTROL PERÍODO: Presentado - Descuento - Saldo", ctrl_periodo],
    ]
    return pd.DataFrame(out, columns=["Concepto", "Importe"])


# ---------------- UI principal ----------------
uploaded = st.file_uploader("Subí un PDF de liquidaciones (Nación Servicios)", type=["pdf"])
if uploaded is None:
    st.info("La app no almacena datos. Procesamiento local en memoria.")
    st.stop()

data = uploaded.read()
txt_full = _text_from_pdf(io.BytesIO(data)).strip()
if not txt_full:
    st.error(
        "No se pudo leer texto del PDF. "
        "Este PDF parece estar escaneado (solo imagen). "
        "La herramienta funciona con PDFs donde el texto sea seleccionable."
    )
    st.stop()

df = process_pdf(data)
if df.empty:
    st.error("No se detectaron páginas 'LIQUIDACION A COMERCIO' con 'Nro. de Liquidación'.")
    st.stop()

# Período (derivado por rango de Fecha Pago)
fechas = pd.to_datetime(df["Fecha Pago"], dayfirst=True, errors="coerce").dropna()
first_date = fechas.min() if not fechas.empty else None
last_date = fechas.max() if not fechas.empty else None

periodo = ""
if first_date is not None and last_date is not None:
    if first_date.month == last_date.month and first_date.year == last_date.year:
        periodo = f"{last_date.strftime('%m/%Y')}"
    else:
        periodo = f"{first_date.strftime('%m/%Y')}–{last_date.strftime('%m/%Y')}"

# ---------------- Control de período ----------------
st.subheader("Control de período (TOTAL PRESENTADO / TOTAL DESCUENTO / SALDO)")

total_presentado = float(df["Total Presentado"].fillna(0).sum())
total_descuento = float(df["Total Descuento"].fillna(0).sum())
total_saldo = float(df["Saldo"].fillna(0).sum())

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("TOTAL PRESENTADO", f"$ {fmt_ar(total_presentado)}")
with c2:
    st.metric("TOTAL DESCUENTO", f"$ {fmt_ar(total_descuento)}")
with c3:
    st.metric("SALDO", f"$ {fmt_ar(total_saldo)}")

ctrl = total_presentado - total_descuento - total_saldo
c4, c5 = st.columns(2)
with c4:
    st.metric("Control (Presentado - Descuento - Saldo)", f"$ {fmt_ar(ctrl)}")
with c5:
    st.metric("Período (derivado)", periodo or "—")

if abs(ctrl) < 0.50:
    st.success("Control OK (tolerancia por redondeos).")
else:
    st.error("Control NO OK: revisá si falta alguna página o si hay descuentos no contemplados.")

# ---------------- Resumen Operativo ----------------
st.subheader("Resumen Operativo: Registración Módulo IVA")

df_ro = resumen_operativo(df)
df_ro_view = df_ro.copy()
df_ro_view["Importe"] = df_ro_view["Importe"].map(fmt_ar)
st.dataframe(df_ro_view, use_container_width=True, hide_index=True)

# Métricas IVA / IIBB
neto_21 = float(df["Gastos Comisiones Neto 21%"].fillna(0).sum())
iva_21 = float(df["IVA 21%"].fillna(0).sum())
ret_iibb = float(df["Ret IIBB"].fillna(0).sum())

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Comisiones (Neto 21%)", f"$ {fmt_ar(neto_21)}")
with m2:
    st.metric("IVA 21%", f"$ {fmt_ar(iva_21)}")
with m3:
    st.metric("Ret IIBB", f"$ {fmt_ar(ret_iibb)}")

# ---------------- Grilla ----------------
st.subheader("Grilla de liquidaciones")

df_view = df.copy()

# Formateo visual
if "Fecha Pago" in df_view.columns:
    # mantenemos string dd/mm/yyyy tal como viene
    pass

money_cols = [
    "Total Presentado",
    "Total Descuento",
    "Saldo",
    "Gastos Comisiones Neto 21%",
    "IVA 21%",
    "Gasto 21% Total (Neto+IVA)",
    "Ret IIBB",
    "Otros descuentos (no clasificados)",
    "Control: Presentado - Descuento - Saldo",
    "Control: Neto - IVA/0.21",
]
for c in money_cols:
    if c in df_view.columns:
        df_view[c] = df_view[c].map(fmt_ar)

st.dataframe(df_view, use_container_width=True, hide_index=True)

# ---------------- Descargas ----------------
st.subheader("Descargas")

date_suffix = ""
if first_date is not None and last_date is not None:
    date_suffix = f"_{first_date.strftime('%Y%m%d')}_{last_date.strftime('%Y%m%d')}"

# Excel
try:
    import xlsxwriter  # noqa: F401

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Detalle")
        df_ro.to_excel(writer, index=False, sheet_name="Resumen_Operativo")

        wb = writer.book
        money_fmt = wb.add_format({"num_format": "#,##0.00"})
        ws = writer.sheets["Detalle"]

        # anchos
        for idx, col in enumerate(df.columns):
            width = min(max(len(str(col)), 12) + 2, 52)
            ws.set_column(idx, idx, width)

        # formato numérico
        for colname in [c for c in df.columns if c in money_cols]:
            j = df.columns.get_loc(colname)
            ws.set_column(j, j, 20, money_fmt)

        # Resumen
        ws2 = writer.sheets["Resumen_Operativo"]
        ws2.set_column(0, 0, 48)
        ws2.set_column(1, 1, 20, money_fmt)

    st.download_button(
        "📥 Descargar Excel",
        data=output.getvalue(),
        file_name=f"ia_nacion_servicios{date_suffix}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
except Exception:
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 Descargar CSV (fallback)",
        data=csv_bytes,
        file_name=f"ia_nacion_servicios{date_suffix}.csv",
        mime="text/csv",
        use_container_width=True,
    )

# PDF Resumen Operativo
if REPORTLAB_OK:
    try:
        pdf_buf = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buf, pagesize=A4, title="Resumen Operativo - Nación Servicios")
        styles = getSampleStyleSheet()

        elems = [
            Paragraph("Resumen Operativo: Nación Servicios", styles["Title"]),
            Paragraph(f"Período: {periodo or '—'}", styles["Normal"]),
            Spacer(1, 10),
        ]

        datos = [["Concepto", "Importe"]]
        for _, r in df_ro.iterrows():
            imp = r["Importe"]
            # si no es número, lo dejamos como texto
            if isinstance(imp, (int, float, np.floating)) and not np.isnan(imp):
                imp_txt = fmt_ar(float(imp))
            else:
                imp_txt = str(imp)
            datos.append([str(r["Concepto"]), imp_txt])

        tbl = Table(datos, colWidths=[360, 140])
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                    ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                ]
            )
        )

        elems.append(tbl)
        elems.append(Spacer(1, 14))
        elems.append(Paragraph("Herramienta para uso interno - AIE San Justo", styles["Normal"]))

        doc.build(elems)

        st.download_button(
            "📄 Descargar PDF – Resumen Operativo",
            data=pdf_buf.getvalue(),
            file_name=f"Resumen_Operativo_Nacion_Servicios{date_suffix}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as e:
        st.info(f"No se pudo generar el PDF del Resumen Operativo: {e}")
