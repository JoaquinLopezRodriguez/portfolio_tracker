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
    """Calcula la TIR (XIRR). Retorna 0 si los flujos no son válidos."""
    if len(cashflows) < 2: return 0.0
    # Validar que haya al menos un flujo positivo y uno negativo
    if all(x >= 0 for x in cashflows) or all(x <= 0 for x in cashflows):
        return 0.0
    try:
        start_date = min(dates)
        def npv(rate):
            return sum([cf / (1 + rate) ** ((d - start_date).days / 365.0) for cf, d in zip(cashflows, dates)])
        return optimize.brentq(npv, -0.99, 10.0)
    except:
        return 0.0

def obtener_metricas_riesgo(serie_cartera, serie_spy, rf=0.04):
    if serie_cartera.empty or len(serie_cartera) < 5 or serie_cartera.iloc[-1] == 0:
        return 0, 0, 0, 0, 0
    
    rets = serie_cartera.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if rets.empty: return 0, 0, 0, 0, 0
    
    vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() * 252 - rf) / vol if vol > 0 else 0
    sortino = (rets.mean() * 252 - rf) / (rets[rets<0].std() * np.sqrt(252)) if len(rets[rets<0])>0 else 0
    
    beta = 1.0
    if not serie_spy.empty and not serie_spy.isna().all():
        spy_rets = serie_spy.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        common = pd.concat([rets, spy_rets], axis=1).dropna()
        if len(common) > 5:
            beta = np.cov(common.iloc[:,0], common.iloc[:,1])[0][1] / (np.var(common.iloc[:,1]) + 1e-9)
            
    var_95 = np.percentile(rets, 5)
    return vol, sharpe, sortino, beta, var_95

@st.cache_data(ttl=3600, show_spinner=False)
def simular_cartera_final(df_movs):
    if df_movs.empty: return None, None, None
    df = df_movs.copy()
    df['fecha'] = pd.to_datetime(df['fecha']).dt.normalize()
    start_date = df['fecha'].min()
    end_date = pd.Timestamp.now().normalize()
    
    activos = [i for i in df['instrumento'].unique() if i != 'CASH']
    tickers = list(set(activos + ['SPY']))
    
    # Descarga de datos
    with st.spinner('Obteniendo datos de mercado...'):
        data = yf.download(tickers, start=start_date - timedelta(days=15), end=end_date, progress=False)
        if 'Close' not in data or data['Close'].empty:
            st.error("Error: No se pudieron descargar precios. Revisa los Tickers.")
            return None, None, None
        precios = data['Close'].ffill().bfill()

    fechas_rango = pd.date_range(start_date, end_date)
    precios = precios.reindex(fechas_rango).ffill().bfill()
    
    hist_cartera, hist_spy = [], []
    posiciones = {a: 0.0 for a in activos}
    caja, pos_spy = 0.0, 0.0
    
    for dia in fechas_rango:
        m_dia = df[df['fecha'] == dia]
        p_spy = precios.loc[dia, 'SPY'] if 'SPY' in precios.columns else 1.0
        
        for _, m in m_dia.iterrows():
            m_tipo, m_monto = str(m['tipo']).upper(), abs(m['monto'])
            if m_tipo == 'DEPOSITO':
                caja += m_monto
                pos_spy += m_monto / p_spy if p_spy > 0 else 0
            elif m_tipo == 'RETIRO':
                caja -= m_monto
                pos_spy -= m_monto / p_spy if p_spy > 0 else 0
            elif m_tipo == 'COMPRA':
                caja -= m_monto
                posiciones[m['instrumento']] += abs(m.get('cantidad', 0))
            elif m_tipo == 'VENTA':
                caja += m_monto
                posiciones[m['instrumento']] -= abs(m.get('cantidad', 0))
            elif m_tipo == 'DIVIDENDO':
                caja += m_monto

        val_mkt = sum(posiciones[a] * precios.loc[dia, a] for a in activos if a in precios.columns)
        hist_cartera.append(val_mkt + caja)
        hist_spy.append(pos_spy * p_spy)

    return pd.Series(hist_cartera, index=fechas_rango), pd.Series(hist_spy, index=fechas_rango), precios

# --- INTERFAZ DE TABS ---
tab1, tab2, tab3 = st.tabs(["🏠 Principal", "📥 Movimientos", "📉 Métricas & Riesgo"])
with tab2:
    st.subheader("📥 Gestión de Datos")
    
    # --- 1. PRIMERO: Inicializamos los datos si no existen ---
    if 'df_movimientos' not in st.session_state:
        st.session_state.df_movimientos = pd.DataFrame([
            {'fecha': pd.to_datetime('2023-01-01'), 'tipo': 'DEPOSITO', 'instrumento': 'CASH', 'monto': 10000.0, 'cantidad': 0}
        ])
    
    if 'df_foto' not in st.session_state:
        st.session_state.df_foto = pd.DataFrame([
            {'instrumento': 'CASH', 'monto': 0.0, 'cantidad': 0}
        ])

    # --- 2. SEGUNDO: Los cargadores de archivos (Uploader) ---
    archivo_h = st.file_uploader("Subir Historial", type=['xlsx', 'csv'], key="u_hist")
    if archivo_h:
        df_h = pd.read_csv(archivo_h) if archivo_h.name.endswith('.csv') else pd.read_excel(archivo_h)
        df_h.columns = df_h.columns.str.lower().str.strip()
        st.session_state.df_movimientos = df_h

    # --- 3. TERCERO: Los editores (Data Editor) ---
    # Ahora que ya existen en session_state, no darán error
    st.session_state.df_movimientos = st.data_editor(
        st.session_state.df_movimientos, num_rows="dynamic", use_container_width=True, key="ed_hist",
        column_config={"fecha": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY")}
    )

    st.divider()

    archivo_f = st.file_uploader("Subir Foto Actual", type=['xlsx', 'csv'], key="u_foto")
    if archivo_f:
        df_f = pd.read_csv(archivo_f) if archivo_f.name.endswith('.csv') else pd.read_excel(archivo_f)
        df_f.columns = df_f.columns.str.lower().str.strip()
        st.session_state.df_foto = df_f

    st.session_state.df_foto = st.data_editor(
        st.session_state.df_foto, num_rows="dynamic", use_container_width=True, key="ed_foto"
    )

    btn_actualizar = st.button("🚀 Actualizar Reporte", use_container_width=True, type="primary")

# PROCESAMIENTO
# 1. Preparamos el lugar donde guardaremos los resultados para que no se borren al navegar
if 'resultados' not in st.session_state:
    st.session_state.resultados = {'curva': None, 'spy': None, 'precios': None, 'v_real': 0}

# --- PROCESAMIENTO HÍBRIDO ---
if btn_actualizar:
    df_h_val = st.session_state.df_movimientos.dropna(subset=['instrumento', 'monto'])
    df_f_val = st.session_state.df_foto.dropna(subset=['instrumento', 'monto'])
    
    if not df_h_val.empty:
        # 1. Reconstruimos el pasado con los movimientos para el gráfico y riesgo
        curva_h, spy_h, precios_h = simular_cartera_final(df_h_val)
        
        # 2. Ajustamos el valor de HOY con la Foto Actual para que sea exacto
        v_real_hoy = df_f_val['monto'].sum() if not df_f_val.empty else curva_h.iloc[-1]
        
        # 3. Guardamos todo. Usamos curva_h para Volatilidad/Sharpe
        st.session_state.resultados = {
            'curva': curva_h, 
            'spy': spy_h, 
            'precios': precios_h, 
            'v_real': v_real_hoy,
            'df_foto': df_f_val
        }
        st.success("¡Reporte sincronizado!")

# Extraemos para las pestañas
res = st.session_state.resultados
curva, spy_c, df_p, v_act = res['curva'], res['spy'], res['precios'], res['v_real']

# El if curva is not None... debe seguir abajo normalmente
if curva is not None and not curva.empty:
    with tab1:
        v_act = v_act_real # <--- LA FOTO ACTUAL MANDA AQUÍ
        df_m = st.session_state.df_movimientos
        
        # El neto invertido SIEMPRE sale de los movimientos (depósitos - retiros)
        dep = df_m[df_m['tipo'].str.upper()=='DEPOSITO']['monto'].sum()
        ret = df_m[df_m['tipo'].str.upper()=='RETIRO']['monto'].sum()
        neto = dep - ret
        
        c1, c2, c3 = st.columns(3)
        # Ahora v_act (foto) vs neto (movimientos) dará el % real sin errores de yfinance
        c1.metric("Valor Actual", f"$ {v_act:,.2f}", f"{(v_act/neto-1):.2%}" if neto > 0 else "0%")
        
        # La TIR también se vuelve más precisa
        cfs_g = df_m[df_m['tipo'].isin(['DEPOSITO','RETIRO'])].apply(lambda x: -abs(x['monto']) if x['tipo'].upper()=='DEPOSITO' else abs(x['monto']), axis=1).tolist() + [v_act]
        fechas_g = pd.to_datetime(df_m[df_m['tipo'].isin(['DEPOSITO','RETIRO'])]['fecha']).tolist() + [pd.Timestamp.now()]
        c2.metric("TIR Cartera", f"{xirr_core(cfs_g, fechas_g):.2%}")
        
        c3.metric("Benchmark (SPY)", f"$ {spy_c.iloc[-1]:,.2f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curva.index, y=curva, name="Mi Cartera", line=dict(color="#00CC96", width=2)))
        fig.add_trace(go.Scatter(x=spy_c.index, y=spy_c, name="SPY Bench", line=dict(color="#EF553B", dash="dot")))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # 1. Métricas
        vol, sha, sor, bet, var = obtener_metricas_riesgo(curva, spy_c)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Volatilidad", f"{vol:.2%}"); m2.metric("Sharpe", f"{sha:.2f}"); m3.metric("Sortino", f"{sor:.2f}"); m4.metric("Beta", f"{bet:.2f}"); m5.metric("VaR 95%", f"{var:.2%}")
        
        st.divider()
        
        # 2. RENDIMIENTO POR ACTIVO (AQUÍ ESTÁ LA CORRECCIÓN)
        st.subheader("Rendimiento por Activo")
        resumen = []
        for act in [a for a in st.session_state.df_movimientos['instrumento'].unique() if a != 'CASH']:
            m_act = st.session_state.df_movimientos[st.session_state.df_movimientos['instrumento'] == act].copy()
            p_final = df_p[act].iloc[-1] if act in df_p.columns else 0
            cant = m_act[m_act['tipo'].str.upper()=='COMPRA']['cantidad'].sum() - m_act[m_act['tipo'].str.upper()=='VENTA']['cantidad'].sum()
            val_act = cant * p_final
            
            # Construir flujos para TIR individual
            cfs_i = []
            for _, row in m_act.iterrows():
                t, m = str(row['tipo']).upper(), abs(row['monto'])
                if t == 'COMPRA': cfs_i.append(-m)
                elif t in ['VENTA', 'DIVIDENDO']: cfs_i.append(m)
            cfs_i.append(val_act) # Valor final como flujo positivo
            
            fechas_i = pd.to_datetime(m_act['fecha']).tolist() + [pd.Timestamp.now()]
            
            resumen.append({
                'Activo': act,
                'Posición': f"{cant:,.2f}",
                'Valor Mercado': f"$ {val_act:,.2f}",
                'TIR (XIRR)': f"{xirr_core(cfs_i, fechas_i):.2%}"
            })
        
        if resumen:
            st.table(pd.DataFrame(resumen))
        
        # 3. Gráficos y Correlación
        c_left, c_right = st.columns(2)
        with c_left:
            st.write("**Drawdown Histórico**")
            dd = (curva - curva.cummax()) / (curva.cummax() + 1e-9)
            st.area_chart(dd, color="#ff4b4b")
        with c_right:
            st.write("**Matriz de Correlación**")
            if not df_p.empty:
                st.plotly_chart(px.imshow(df_p.pct_change().corr(), text_auto=".2f", color_continuous_scale='RdBu_r'), use_container_width=True)
else:
    st.info("Carga datos válidos para ver el análisis.")



