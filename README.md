# Paper trading para agentes de trading

Un entorno para probar agentes de trading (de IA o clásicos) **con dinero ficticio** y medir
si realmente le ganan a la alternativa aburrida, que es comprar y mantener, **después de costos**.

- **Dos brokers:** un simulador local que no necesita cuenta, o tu cuenta **paper** de Alpaca
  (el dinero real está bloqueado en el código).
- **Dos referencias:** el benchmark (SPY por defecto) y *comprar y mantener los mismos tickers*.
- **Costos incluidos:** comisiones, slippage y el costo estimado de las llamadas al LLM.
- **Métricas:** retorno total y anualizado, volatilidad, Sharpe y caída máxima.
- **Agentes incluidos:** `buy_and_hold`, `sma_crossover` (ejemplo clásico) y un adaptador para
  [TradingAgents](https://github.com/TauricResearch/TradingAgents).

## Reglas para no engañarse (ya vienen integradas)

1. **Se decide con el cierre de hoy y se ejecuta a la apertura de mañana.** Nunca se opera al
   mismo precio que generó la señal.
2. **La estrategia solo ve datos hasta el día de la decisión**, así que no puede mirar al futuro.
3. **Los backtests de estrategias con LLM están bloqueados** (se pueden forzar con `--allow-llm`).
   El modelo se entrenó con datos posteriores a esas fechas, así que ya "sabe" lo que pasó. Para
   un agente LLM, la única prueba válida es paper trading hacia adelante.
4. **Cada decisión queda registrada con la fecha y hora en que se tomó** (`decisions.jsonl`). Ese
   archivo no se reescribe, así que no se puede maquillar el historial.
5. **No hay veredicto con menos de ~6 meses** (126 días hábiles) de datos.
6. **Siempre hay dos comparaciones.** Si una estrategia le gana a SPY pero no a comprar y mantener
   sus propias acciones, su "ventaja" viene de qué acciones se eligieron, no de cómo opera.
7. **Kill switch opcional:** si la caída desde el máximo pasa el límite, vende todo y se detiene.

### Un ejemplo real de por qué importa la regla 6

Backtest de `configs/sma_ejemplo.toml` (AAPL, MSFT y NVDA, de 2022-01-03 a 2026-10-16):

```
                                Estrategia      Mismos tickers     Benchmark (SPY)
Retorno total                     +112.08%            +257.17%             +70.23%
Sharpe                                0.86                0.95                0.73
Caída máxima                       -24.26%             -39.21%             -24.50%

Después de costos:
  vs Benchmark (SPY): +41.85 puntos porcentuales -> SÍ le gana
  vs comprar y mantener los mismos tickers: -145.09 puntos porcentuales -> NO le gana
Veredicto: NO agrega valor. Comprar y mantener habría dado más.
```

Parece que "le gana al mercado", pero solo porque elegimos NVDA sabiendo ya que se disparó.
Haber comprado esas tres acciones sin hacer nada habría dado más del doble.

## Instalación

Requiere Python 3.11 o más reciente.

```bash
git clone https://github.com/pelukin42/trading.git
cd trading
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[yfinance,dev]"
```

Todos los comandos se corren desde la carpeta del repo.

## Uso rápido (sin cuentas, con datos de Yahoo Finance)

```bash
# 1. Backtests (solo para estrategias sin LLM)
papertrade backtest configs/benchmark_spy.toml --start 2022-01-01
papertrade backtest configs/sma_ejemplo.toml --start 2022-01-01

# 2. Paper trading hacia adelante: una vez por día hábil, DESPUÉS del cierre
#    (4:00 pm hora de Nueva York)
papertrade run configs/sma_ejemplo.toml

# 3. Ver cómo va contra el benchmark
papertrade report configs/sma_ejemplo.toml
papertrade report configs/sma_ejemplo.toml --backtest   # el último backtest
```

Si corres `run` dos veces el mismo día, no vuelve a decidir. Si te saltas días con el broker
simulado, no pasa nada: las posiciones se mantienen y el agente decide la próxima vez que lo corras.

## Usar tu cuenta paper de Alpaca

1. Crea una cuenta en [alpaca.markets](https://alpaca.markets). Para paper trading no necesitas
   depositar dinero.
2. En el panel, cambia a **Paper**, genera tus **API Keys** y cópialas:
   ```bash
   cp .env.example .env    # y pega ALPACA_API_KEY y ALPACA_SECRET_KEY
   ```
3. Usa una **cuenta paper dedicada** a cada experimento. El agente administra *toda* la cuenta y
   vende lo que no esté en `symbols`. Alpaca permite tener varias cuentas paper.
4. Corre `papertrade run configs/sma_alpaca.toml` una vez por día hábil, después del cierre. Las
   órdenes de mercado se encolan y se ejecutan a la apertura siguiente, igual que en el simulador.

Limitaciones de Alpaca paper (según [su documentación](https://docs.alpaca.markets/docs/paper-trading)):
no simula dividendos ni impacto de mercado, y las cuentas solo-paper usan datos IEX. Por eso, en
modo Alpaca el benchmark se compara **sin dividendos**.

## Probar TradingAgents

```bash
git clone https://github.com/TauricResearch/TradingAgents.git
pip install ./TradingAgents
```

1. Pon en `.env` la llave de tu proveedor de LLM (por ejemplo `ANTHROPIC_API_KEY` u `OPENAI_API_KEY`).
2. En `configs/tradingagents.toml`, ajusta `[strategy.config]` (proveedor y modelos) y
   `cost_per_call_usd`, que es lo que te cuesta analizar un ticker una vez. Revisa tu factura después
   de las primeras corridas y corrige el número.
3. Córrelo a diario con `papertrade run configs/tradingagents.toml`. Solo consulta al agente una vez
   por semana (`rebalance = "weekly"`) para limitar el gasto.

Cómo se traducen sus ratings: cada ticker tiene un cupo de 1/N del capital.
`Buy` = 100% del cupo, `Overweight` = 75%, `Underweight` = 25% y `Sell` = 0%.
`Hold`, o un rating que no se pueda leer, mantienen la posición actual.

## Correrlo automáticamente todos los días

**Linux / Mac (cron).** Ajusta la hora para que equivalga a las ~4:30 pm de Nueva York en tu zona horaria:

```cron
30 16 * * 1-5  cd /ruta/a/trading && .venv/bin/papertrade run configs/sma_ejemplo.toml >> runs/cron.log 2>&1
```

**Windows (Programador de tareas).** Crea una tarea diaria de lunes a viernes:
- Programa: `C:\ruta\a\trading\.venv\Scripts\papertrade.exe`
- Argumentos: `run configs\sma_ejemplo.toml`
- Iniciar en: `C:\ruta\a\trading`

## Conectar tu propio agente

```python
# papertrade/strategies.py
class MiAgente(Strategy):
    uses_llm = True   # si usa un LLM: bloquea backtests contaminados

    def decide(self, as_of, data, current):
        # data.close / data.open: precios hasta `as_of` (filas = días, columnas = tickers)
        # current: pesos actuales, p. ej. {"AAPL": 0.3}
        pesos = {s: 1 / len(self.symbols) for s in self.symbols}
        return Decision(pesos, notes={"motivo": "..."}, llm_cost_usd=0.0)

STRATEGIES["mi_agente"] = MiAgente
```

Después, en tu configuración, pon `type = "mi_agente"` dentro de `[strategy]`. Cualquier otra clave
de esa sección se pasa como argumento a `__init__`.

## Configuración

Cada experimento es un archivo TOML en `configs/`. El más comentado es
[`configs/sma_ejemplo.toml`](configs/sma_ejemplo.toml). Las claves principales:

| Clave | Qué hace |
|---|---|
| `name` | Nombre del experimento; sus archivos quedan en `runs/<name>/` |
| `symbols`, `benchmark` | Tickers que opera el agente y contra qué se compara |
| `rebalance` | Cada cuánto se consulta al agente: `daily`, `weekly` o `monthly` |
| `[broker] type` | `sim` (local) o `alpaca` (cuenta paper) |
| `[broker] slippage_bps`, `commission_*` | Costos de ejecución en el simulador |
| `[data] source` | `yfinance`, `alpaca` o `csv` (archivos `<TICKER>.csv` con `date,open,close`) |
| `[risk] max_weight` | Peso máximo por ticker |
| `[risk] max_drawdown_stop` | Kill switch (0 = desactivado) |

## Archivos que genera

```
runs/<name>/
  decisions.jsonl   # cada decisión del agente, con fecha y hora de registro (no se reescribe)
  equity.csv        # valor del portafolio por día (+ precios de referencia)
  fills.csv         # ejecuciones simuladas con comisión y slippage (broker sim)
  orders.csv        # órdenes enviadas a Alpaca (broker alpaca)
  backtest/         # resultados del último backtest (se regenera; está en .gitignore)
```

Con el broker simulado, `equity.csv` y `fills.csv` se reconstruyen en cada corrida a partir de
`decisions.jsonl` y de precios actualizados (ajustados por splits y dividendos). Con Alpaca,
`equity.csv` guarda lo que reporta la cuenta cada día.

## Pruebas

```bash
pytest
```

## Aviso

Este proyecto es educativo. No es asesoría financiera. Un buen resultado en paper trading no
garantiza resultados con dinero real.
