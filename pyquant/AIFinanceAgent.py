from llama_index.llms.gemini import Gemini
from llama_index.core import Settings
from llama_index.tools.code_interpreter.base import CodeInterpreterToolSpec
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.google_genai import GoogleGenAI

import yfinance as yf
from datetime import datetime
import os
import asyncio
import matplotlib

import io, contextlib, subprocess, sys
import contextlib


# Set up Gemini API key
api_key = os.getenv("GOOGLE_API_KEY")

def get_stock_data(ticker: str, period: str = "1mo"):
    data = yf.Ticker(ticker).history(period=period)
    return data.to_csv()

# Tool: get today's date
def get_today_date():
    return str(datetime.today().date())

def run_generated_code(code_str: str) -> str:
    """
    Executes code_str in an isolated namespace,
    installs any missing modules on the fly,
    and returns the captured stdout.
    """
    namespace = {}
    buffer = io.StringIO()

    def _exec():
        exec(code_str, namespace)

    with contextlib.redirect_stdout(buffer):
        try:
            _exec()
        except ModuleNotFoundError as e:
            pkg = e.name
            print(f"[Runner] Module not found: {pkg}. Installing…")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            # Clear any partial output
            buffer.truncate(0); buffer.seek(0)
            _exec()

    return buffer.getvalue()

# Async tool retriever (required by FunctionAgent)
async def get_tools():
    tools =  [
        FunctionTool.from_defaults(fn=get_stock_data),
        FunctionTool.from_defaults(fn=get_today_date),
        *CodeInterpreterToolSpec().to_tool_list()
    ]
    print("Registered tools:", [t.name for t in tools])  # Debug print
    return tools


# Async main function
async def main():
    llm = GoogleGenAI(model="models/gemini-2.0-flash")

    agent = FunctionAgent(
        llm=llm,
        retrieve_tools=get_tools,
        system_prompt = (
                "You are an intelligent trading agent whose only job is to *generate* a valid, runnable Python script.  \n"
    "The script must use only pure-Python libraries—no C-compiled dependencies like TA-Lib.  \n"
    "IMPORTANT: In any pattern‑detection or crossover logic, you must compare only Python scalars, never entire pandas Series.  \n"
    "To do this, iterate over rows (e.g. with `for row in df.itertuples():`) or use explicit `.iloc`/`.iat` indexing.  \n"
    "Avoid `if df['Close'] > df['Open']:` style tests.  \n"
    "You may use `yfinance`, `pandas`, `numpy`, `pandas_ta`, `matplotlib`, and any stdlib.  \n"
    "Ensure the code runs with no errors  \n"
    "Do NOT execute any code or call any tools—only output the code."
        )
    )

    prompt = (
    "Write a self-contained Python script that, given a 14-day candlestick chart (or fetching OHLCV via yfinance), does the following:\n"
    "1. Load or fetch the last 14 days of OHLCV data for a ticker.\n"
    "2. Compute 5-day and 10-day moving averages.\n"
    "3. Identify common candlestick patterns (hammer, doji, engulfing, morning star, etc.) and label bullish/bearish signals.\n"
    "4. Evaluate moving average crossovers and momentum (e.g., slope or ROC).\n"
    "5. Assess volume trends alongside price moves.\n"
    "6. Determine support and resistance from recent highs/lows.\n"
    "7. Based on that analysis, print whether there is a potential 3-5% gain setup in the next 5-7 trading days.\n"
    "   • If yes, calculate and print an optimal buy price, profit target, and stop-loss (risk/reward ≥ 2:1).\n"
    "   • If no, print an explanation of why no short-term trade is recommended.\n"
    "Your output should be valid, runnable Python code, with comments explaining each step."
)
    response = await agent.run(prompt)
    #print(type(response))
    #print(response)

    generated_code = str(response).replace("```python", "")
    generated_code = generated_code.replace("```", "")

    print(generated_code)
    results=run_generated_code(generated_code)
    print(results)

# Entry point
if __name__ == "__main__":
    asyncio.run(main())