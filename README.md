# agentrade — trading agéntico con control de riesgo

Un agente de IA (Claude) analiza el mercado y decide; un **motor de riesgo determinista** valida cada
orden antes de que llegue al broker; todo queda registrado con su justificación.

```
          ┌──────────────┐  herramientas  ┌──────────────┐   única vía   ┌──────────────┐
 Mercado ─▶│  Agente LLM  │───────────────▶│ Motor riesgo │──────────────▶│    Broker    │
          │ (Claude API) │◀── razones de ─│ (código puro)│               │ Alpaca / Sim │
          └──────┬───────┘    rechazo     └──────┬───────┘               └──────────────┘
                 │                               │
                 └────────── Diario SQLite (decisiones, órdenes, tokens) ─┘
```

**Principio de diseño:** el modelo propone, el código dispone. El agente nunca toca el broker
directamente: cada orden pasa por `OrderRouter → RiskManager`, que aplica límites duros que el
modelo no puede negociar ni saltarse.

## Qué incluye

| Componente | Archivo | Qué hace |
|---|---|---|
| Agente | `src/agentrade/agent/` | Loop de herramientas con Claude: portafolio, datos de mercado, órdenes, cierre. Reporte por ciclo que alimenta al siguiente. Búsqueda web de noticias opcional. |
| Riesgo | `src/agentrade/risk.py` | Tamaño máximo por posición, exposición bruta, nº de posiciones, tope por orden, pérdida diaria máxima, órdenes por día, stop-loss obligatorio, sin cortos, watchlist, kill switch. Reducir riesgo siempre está permitido. |
| Brokers | `src/agentrade/broker/` | **Cuenta de práctica local** (por defecto: sin registrarte en ningún broker, precios reales de Yahoo Finance, horario y feriados oficiales de NYSE, estado guardado en `data/paper_account.json`), Alpaca (paper y live, órdenes OTO con stop GTC) y un simulador de backtests sin sesgo de anticipación. |
| Backtesting | `src/agentrade/backtest.py` | Mismo router y mismo riesgo que en vivo. Métricas: CAGR, Sharpe, drawdown, tasa de acierto, salidas por stop, contra buy & hold. |
| Estrategias base | `src/agentrade/strategies.py` | `sma_cross`, `momentum`, `equal_weight`: la vara que el agente debe superar para justificar su costo. |
| Diario | `src/agentrade/journal.py` | Cada ciclo, llamada a herramienta, resumen de razonamiento, veredicto de riesgo y orden. |

## Puesta en marcha

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env       # basta con ANTHROPIC_API_KEY; Alpaca es opcional
pytest -q                  # corre sin red ni claves
```

### 1. Backtest de estrategias (gratis, sin claves)

```bash
agentrade backtest --strategy sma_cross                              # datos sintéticos
agentrade backtest --strategy momentum --source yahoo --days 1000    # precios reales
agentrade backtest --strategy sma_cross --source csv:./mis_datos     # AAPL.csv con date,open,high,low,close,volume
```

### 2. Backtest del agente (consume API de Claude)

```bash
agentrade backtest --strategy agent --source synthetic --every 5 --max-cycles 20
```

Decide una vez por semana (`--every 5`) durante 20 ciclos e imprime el costo acumulado estimado.
Con datos sintéticos el modelo no puede "recordar" el futuro; con datos reales de tickers conocidos
sí podría tener sesgo por su entrenamiento, así que trata esos resultados con escepticismo.

### 3. Operar en práctica (paper)

Por defecto (`AGENTRADE_BROKER=local`) la plataforma lleva su propia cuenta de práctica con US$100.000
virtuales: no necesitas cuenta en ningún broker. Las órdenes a mercado se ejecutan al precio actual si
NYSE está abierto, o en la apertura de la siguiente sesión si está cerrado; los stops se vigilan con el
precio en vivo y con los mínimos diarios.

```bash
agentrade run --once     # un ciclo y salir (ideal para cron)
agentrade run            # bucle: cada AGENTRADE_CYCLE_MINUTES con el mercado abierto
agentrade status         # cuenta, posiciones, órdenes pendientes, kill switch
agentrade journal -n 5   # qué decidió el agente y por qué
agentrade kill --flatten # detener todo y cerrar posiciones
agentrade resume
agentrade reset-paper --cash 100000 --yes   # empezar de cero
```

Para usar una cuenta paper de Alpaca en su lugar: `AGENTRADE_BROKER=alpaca` más sus claves en `.env`.

### 4. Pasar a dinero real

Solo después de semanas en paper con resultados que superen a las estrategias base. Requiere un broker
real (Alpaca):

```bash
AGENTRADE_BROKER=alpaca
AGENTRADE_MODE=live
AGENTRADE_LIVE_CONFIRM=I_UNDERSTAND_REAL_MONEY_IS_AT_RISK
```

Empieza con una fracción pequeña del capital y límites más estrictos que los de paper.

## Costos del agente

Por defecto usa `claude-opus-5-5` con esfuerzo `medium`. Un ciclo típico son 4–10 llamadas con
caché de prompt; el costo estimado de cada ciclo se imprime y queda en el diario. Para reducirlo:
`AGENTRADE_EFFORT=low`, `AGENTRADE_MODEL=claude-sonnet-5-5`, menos ciclos por día o una watchlist
más corta. Las respuestas que el filtro de seguridad rechace por error se reintentan en el servidor
con un modelo alternativo (`fallbacks: "default"`).

## Advertencias

- Ningún sistema garantiza ganancias. En los backtests sintéticos incluidos, las estrategias simples
  **no** le ganan a buy & hold. El objetivo del agente es demostrar, con datos, que agrega valor
  ajustado por riesgo; si no lo logra, no debe operar con dinero real.
- La integración con Alpaca está probada contra respuestas simuladas de su API; valídala primero
  en tu cuenta paper.
- Los datos de Yahoo Finance son no oficiales y para uso personal: sus términos no permiten
  redistribuirlos. Para vender la plataforma hace falta un proveedor de datos con licencia.
- La cuenta de práctica local usa barras diarias, no intradía: un stop que se toca y se recupera
  entre dos revisiones solo se detecta al cerrar el día (por el mínimo diario).
- La cuenta de práctica local aplica splits (por evento o detectando que Yahoo reescribió el
  historial) y acredita dividendos en la fecha ex-dividendo. Si pasa mucho tiempo sin abrirse, al
  volver reproduce día por día lo que ocurrió: órdenes en cola, stops y dividendos.
- Venderlo a terceros cambia el marco legal: si la plataforma opera o recomienda sobre el dinero de
  otros, puede requerir registro ante el regulador de valores de cada país. Consulta a un abogado
  antes de cobrar.

## Hoja de ruta

1. **Validación (yo como usuario):** semanas en paper, comparar el agente contra `sma_cross` y
   `momentum` con el diario; ajustar prompts, límites y watchlist.
2. **Más poder:** cripto 24/7 (Alpaca crypto), barras intradía, opciones con límites de griegas,
   múltiples agentes especializados (analista, gestor de riesgo, ejecutor), evaluaciones
   automáticas de calidad de decisiones.
3. **Producto vendible:** panel web (equity, posiciones, razonamiento de cada trade), multiusuario
   con claves de broker cifradas por cliente, facturación por suscripción, alertas (Telegram/email),
   y registro de auditoría exportable.
