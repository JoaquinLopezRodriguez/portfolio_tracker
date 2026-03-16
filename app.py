import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy import optimize

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Portfolio Tracker", layout="wide")

# --- PORTADA / ENCABEZADO ---
st.title("📊 My Portfolio Management")
st.markdown("### Reporte de rendimiento & risk assessment")
st.caption(f"Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
st.divider()

# --- FUNCIONES FINANCIERAS ROBUSTAS ---
def xirr_core(cashflows, dates):
    if len(cashflows) < 2: return 0.0
    if all(x >= 0 for x in cashflows) or all(x <= 0 for x in cashflows): return 0.0
    try:
        start_date = min(dates)
        def npv(rate):
            return sum([cf / (1 + rate) ** ((d - start_date).days / 365.0) for cf, d in zip(cashflows, dates)])
        return optimize.brentq(npv, -0.99, 10.0)
    except: return 0.0

def obtener_metricas_riesgo(serie_cartera, serie_spy, rf=0.04):
    if serie_cartera is None or serie_cartera.empty or len(serie_cartera) < 5:
        return 0, 0, 0, 0, 0
    rets = serie_cartera.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if rets.empty: return 0, 0, 0, 0, 0
    vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() * 252 - rf) / vol if vol > 0 else 0
    sortino = (rets.mean() * 252 - rf) / (rets[rets<0].std() * np.sqrt(252)) if len(rets[rets<0])>0 else 0
    beta = 1.0
    if serie_spy is not None and not serie_spy.empty:
        spy_rets = serie_spy.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        common = pd.concat([rets, spy_rets], axis=1).dropna()
        if len(common) > 5:
            beta = np.cov(common.iloc[:,0], common.iloc[:,1])[0][1] / (np.var(common.iloc[:,1]) + 1e-9)
    return vol, sharpe, sortino, beta, np.percentile(rets, 5)

@st.cache_data(ttl=3600, show_spinner=False)
def simular_cartera_final(df_movs):
    if df_movs.empty: return None, None, None
    df = df_movs.copy()
    df['fecha'] = pd.to_datetime(df['fecha']).dt.normalize()
    start_date, end_date = df['fecha'].min(), pd.Timestamp.now().normalize()
    activos = [i for i in df['instrumento'].unique() if i != 'CASH']
    tickers = list(set(activos + ['SPY']))
    with st.spinner('Obteniendo datos de mercado...'):
        data = yf.download(tickers, start=start_date - timedelta(days=15), end=end_date, progress=False)
        if 'Close' not in data or data['Close'].empty: return None, None, None
        precios = data['Close'].ffill().bfill()
    fechas_rango = pd.date_range(start_date, end_date)
    precios = precios.reindex(fechas_rango).ffill().bfill()
    hist_cartera, hist_spy, posiciones, caja, pos_spy = [], [], {a: 0.0 for a in activos}, 0.0, 0.0
    for dia in fechas_rango:
        m_dia = df[df['fecha'] == dia]
        p_spy = precios.loc[dia, 'SPY'] if 'SPY' in precios.columns else 1.0
        for _, m in m_dia.iterrows():
            tipo, monto = str(m['tipo']).upper(), abs(m['monto'])
            if tipo == 'DEPOSITO': caja += monto; pos_spy += monto/p_spy
            elif tipo == 'RETIRO': caja -= monto; pos_spy -= monto/p_spy
            elif tipo == 'COMPRA': caja -= monto; posiciones[m['instrumento']] += abs(m.get('cantidad', 0))
            elif tipo == 'VENTA': caja += monto; posiciones[m['instrumento']] -= abs(m.get('cantidad', 0))
            elif tipo == 'DIVIDENDO': caja += monto
        val_mkt = sum(posiciones[a] * precios.loc[dia, a] for a in activos if a in precios.columns)
        hist_cartera.append(val_mkt + caja); hist_spy.append(pos_spy * p_spy)
    return pd.Series(hist_cartera, index=fechas_rango), pd.Series(hist_spy, index=fechas_rango), precios

# --- INICIALIZACIÓN DE ESTADO (DATOS DE EJEMPLO) ---
if 'df_movimientos' not in st.session_state:
    st.session_state.df_movimientos = pd.DataFrame([
        {'fecha': pd.to_datetime('2024-01-10'), 'tipo': 'DEPOSITO', 'instrumento': 'CASH', 'monto': 15000.00, 'cantidad': 0},
        {'fecha': pd.to_datetime('2024-01-15'), 'tipo': 'COMPRA', 'instrumento': 'AAPL', 'monto': 4500.00, 'cantidad': 25},
        {'fecha': pd.to_datetime('2024-02-05'), 'tipo': 'COMPRA', 'instrumento': 'MSFT', 'monto': 5000.00, 'cantidad': 12},
        {'fecha': pd.to_datetime('2024-03-20'), 'tipo': 'COMPRA', 'instrumento': 'MELI', 'monto': 3000.00, 'cantidad': 2},
        {'fecha': pd.to_datetime('2024-05-10'), 'tipo': 'DIVIDENDO', 'instrumento': 'AAPL', 'monto': 25.50, 'cantidad': 0},
        {'fecha': pd.to_datetime('2024-06-01'), 'tipo': 'COMPRA', 'instrumento': 'AAPL', 'monto': 1800.00, 'cantidad': 10}
    ])

if 'df_foto' not in st.session_state:
    st.session_state.df_foto = pd.DataFrame([
        {'instrumento': 'CASH', 'monto': 725.50, 'cantidad': 0},
        {'instrumento': 'AAPL', 'monto': 8225.00, 'cantidad': 35},
        {'instrumento': 'MSFT', 'monto': 4950.00, 'cantidad': 12},
        {'instrumento': 'MELI', 'monto': 4100.00, 'cantidad': 2}
    ])

if 'resultados' not in st.session_state:
    # Para que la app no arranque vacía, podemos procesar los defaults al inicio
    st.session_state.resultados = {'curva': None, 'spy': None, 'precios': None, 'v_real': 0.0}
# --- TABS ---
tab1, tab2, tab3 = st.tabs(["🏠 Principal", "📥 Movimientos", "📉 Métricas & Riesgo"])

with tab2:
    st.subheader("📥 Gestión de Datos")
    archivo_h = st.file_uploader("Subir Historial", type=['xlsx', 'csv'], key="u_hist")
    if archivo_h:
        df_h = pd.read_csv(archivo_h) if archivo_h.name.endswith('.csv') else pd.read_excel(archivo_h)
        df_h.columns = df_h.columns.str.lower().str.strip()
        st.session_state.df_movimientos = df_h
    st.session_state.df_movimientos = st.data_editor(st.session_state.df_movimientos, num_rows="dynamic", use_container_width=True, key="ed_hist", column_config={"fecha": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY")})
    
    st.divider()
    
    archivo_f = st.file_uploader("Subir Foto Actual", type=['xlsx', 'csv'], key="u_foto")
    if archivo_f:
        df_f = pd.read_csv(archivo_f) if archivo_f.name.endswith('.csv') else pd.read_excel(archivo_f)
        df_f.columns = df_f.columns.str.lower().str.strip()
        st.session_state.df_foto = df_f
    st.session_state.df_foto = st.data_editor(st.session_state.df_foto, num_rows="dynamic", use_container_width=True, key="ed_foto")
    btn_actualizar = st.button("🚀 Actualizar Reporte", use_container_width=True, type="primary")

# --- PROCESAMIENTO GLOBAL ---
if btn_actualizar:
    df_h_val = st.session_state.df_movimientos.dropna(subset=['instrumento', 'monto'])
    df_f_val = st.session_state.df_foto.dropna(subset=['instrumento', 'monto'])
    if not df_h_val.empty:
        curva_h, spy_h, precios_h = simular_cartera_final(df_h_val)
        v_real = df_f_val['monto'].sum() if not df_f_val.empty else (curva_h.iloc[-1] if curva_h is not None else 0.0)
        st.session_state.resultados = {'curva': curva_h, 'spy': spy_h, 'precios': precios_h, 'v_real': v_real}
        st.success("¡Reporte actualizado!")

# EXTRAER RESULTADOS
res = st.session_state.resultados
curva, spy_c, df_p, v_act_real = res['curva'], res['spy'], res['precios'], res['v_real']

# --- VISTAS ---
if curva is not None and not curva.empty:
    with tab1:
        df_m = st.session_state.df_movimientos
        neto = df_m[df_m['tipo'].str.upper()=='DEPOSITO']['monto'].sum() - df_m[df_m['tipo'].str.upper()=='RETIRO']['monto'].sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Valor Actual Real", f"$ {v_act_real:,.2f}", f"{(v_act_real/neto-1):.2%}" if neto > 0 else "0%")
        cfs_g = df_m[df_m['tipo'].isin(['DEPOSITO','RETIRO'])].apply(lambda x: -abs(x['monto']) if x['tipo'].upper()=='DEPOSITO' else abs(x['monto']), axis=1).tolist() + [v_act_real]
        fechas_g = pd.to_datetime(df_m[df_m['tipo'].isin(['DEPOSITO','RETIRO'])]['fecha']).tolist() + [pd.Timestamp.now()]
        c2.metric("TIR Global (XIRR)", f"{xirr_core(cfs_g, fechas_g):.2%}")
        c3.metric("Benchmark (SPY)", f"$ {spy_c.iloc[-1]:,.2f}")
        fig = go.Figure([go.Scatter(x=curva.index, y=curva, name="Mi Cartera", line=dict(color="#00CC96", width=2)), go.Scatter(x=spy_c.index, y=spy_c, name="SPY Bench", line=dict(color="#EF553B", dash="dot"))])
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        vol, sha, sor, bet, var = obtener_metricas_riesgo(curva, spy_c)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Volatilidad", f"{vol:.2%}"); m2.metric("Sharpe", f"{sha:.2f}"); m3.metric("Sortino", f"{sor:.2f}"); m4.metric("Beta", f"{bet:.2f}"); m5.metric("VaR 95%", f"{var:.2%}")
        st.divider()
        st.subheader("Rendimiento por Activo")
        resumen = []
        df_f_uso = st.session_state.df_foto.dropna(subset=['instrumento', 'monto'])
        for _, row in df_f_uso.iterrows():
            act = row['instrumento']
            if act == 'CASH': continue
            m_act = st.session_state.df_movimientos[st.session_state.df_movimientos['instrumento'] == act]
            # Cálculo de flujos: Compras (-), Ventas (+), Dividendos (+)
            flujos_hist = m_act.apply(lambda x: -abs(x['monto']) if x['tipo'].upper()=='COMPRA' else abs(x['monto']), axis=1).tolist()
            lista_para_calculo = flujos_hist + [row['monto']]
            
            # Resultado No Realizado ($)
            resultado_no_realizado = sum(lista_para_calculo)
            
            # TIR Individual
            fechas_i = pd.to_datetime(m_act['fecha']).tolist() + [pd.Timestamp.now()]
            tir_val = xirr_core(lista_para_calculo, fechas_i)
            
            resumen.append({
                'Activo': act, 
                'Posición': f"{row['cantidad']:,.2f}", 
                'Valor Mercado': f"$ {row['monto']:,.2f}", 
                'Resultado No Realizado ($)': f"$ {resultado_no_realizado:,.2f}",
                'TIR (XIRR)': f"{tir_val:.2%}"
            })
        if resumen: st.table(pd.DataFrame(resumen))
        col_l, col_r = st.columns(2)
        with col_l:
            st.write("**Drawdown Histórico**")
            st.area_chart((curva - curva.cummax()) / (curva.cummax() + 1e-9))
        with col_r:
            st.write("**Matriz de Correlación**")
            if not df_p.empty: st.plotly_chart(px.imshow(df_p.pct_change().corr(), text_auto=".2f", color_continuous_scale='RdBu_r'), use_container_width=True)
else:
    st.info("Carga tus movimientos históricos y posiciones actuales para, y presiona 'Actualizar Reporte' en la pestaña 'Movimientos'. Caso contrario se mostrarán datos de prueba")
