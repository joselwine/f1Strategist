"""
Minimal CLI chatbot for F1 pit-strategy what-if + explanations.

Run:
  python -m app.chatbot_cli
or
  python app/chatbot_cli.py

Assumes your project structure is like:
  Dissertation/
    app/
      chatbot_cli.py
      services/
        strategy_service.py   (we'll create next)
    ML/
      src/
        simulator.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.services.strategy_service import StrategyService


# ----------------------------
# Conversation State
# ----------------------------
@dataclass
class ChatState:
    user_name: Optional[str] = None
    race_id: Optional[str] = None
    driver: Optional[str] = None
    lap: Optional[int] = None  # current lap context, optional
    pending_intent: Optional[str] = None
    pending_text: Optional[str] = None


# ----------------------------
# Intent parsing (rule-based)
# ----------------------------
def parse_pit_lap(text: str) -> Optional[int]:
    """
    Extract a pit lap from user text.
    Examples:
      'what if leclerc pits on lap 25' -> 25
      'pit lap 30' -> 30
    """
    m = re.search(r"(?:lap|l)\s*(\d{1,2})", text.lower())
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None

def parse_driver_name(text: str) -> Optional[str]:
    t = text.lower()
    if "verstappen" in t or "max" in t:
        return "VER"
    if "leclerc" in t or "charles" in t:
        return "LEC"
    if "hamilton" in t or "lewis" in t:
        return "HAM"
    # add more mappings as you like
    return None


def parse_race_alias(text: str) -> Optional[str]:
    t = text.lower()
    # map natural names -> your RaceId format
    if "monaco" in t:
        return "2025-Monaco"
    if "abu dhabi" in t or "abudhabi" in t:
        return "2024-Abu Dhabi"
    # add more mappings as you like
    return None

def parse_driver_code(text: str) -> Optional[str]:
    # very simple: look for a 3-letter all-caps driver code like VER/LEC/HAM
    m = re.search(r"\b([A-Z]{3})\b", text.strip())
    return m.group(1) if m else None

def parse_why_did_pit(text: str) -> tuple[Optional[str], Optional[int]]:
    """
    Examples:
      'why did VER pit' -> ('VER', None)
      'why did verstappen pit on lap 30' (won't match VER unless you add name mapping)
      'why did VER pit on lap 30' -> ('VER', 30)
    """
    driver = parse_driver_code(text)
    lap = parse_pit_lap(text)  # you already have parse_pit_lap
    return driver, lap

def detect_intent(text: str) -> str:
    t = text.lower().strip()

    # basic commands
    if t in {"exit", "quit", "q"}:
        return "exit"
    if t.startswith("set "):
        return "set"
    if t in {"help", "h", "?"}:
        return "help"

    # intents
    if "what if" in t or "simulate" in t:
        return "simulate"
    if "should i pit" in t or "should we pit" in t or "recommend" in t:
        return "recommend"
    if t.startswith("why") or "explain" in t:
        return "explain"
    if "status" in t or "where is" in t or "position" in t or "gap" in t:
        return "status"
    if t.startswith("why did") and "pit" in t:
        return "why_real"
    if "why did" in t and "pit" in t:
        return "real_pit_why"
    if "pitted" in t and ("what if" in t or "if" in t):
        return "simulate"
    if "should" in t and "pit" in t:
        return "recommend"



    # fallback
    return "fallback"


# ----------------------------
# CLI helpers
# ----------------------------
def print_help() -> None:
    print(
        "\nCommands:\n"
        "  set race <RaceId>        e.g. set race 2024-Abu Dhabi\n"
        "  set driver <CODE>        e.g. set driver LEC\n"
        "  set lap <N>              e.g. set lap 25\n"
        "  status                   show current state\n"
        "  what if pit lap <N>      run a what-if simulation\n"
        "  should we pit            recommendation (compares candidate laps)\n"
        "  why                      explanation of last recommendation/sim\n"
        "  help                     show this help\n"
        "  exit                     quit\n"
    )


def ensure_name(state: ChatState) -> None:
    if state.user_name is None:
        name = input("Hi! What’s your name? ").strip()
        if not name:
            name = "friend"
        state.user_name = name
        print(f"Nice to meet you, {state.user_name}! \n")

def require_context(state: ChatState) -> List[str]:
    missing = []
    if not state.race_id:
        missing.append("race_id")
    if not state.driver:
        missing.append("driver")
    return missing


def get_bot(bot: Optional[StrategyService]) -> StrategyService:
    if bot is None:
        print("Loading strategy engine... (models + data)")
        bot = StrategyService()
        print("✅ Strategy engine loaded.\n")
    return bot

def extract_driver_from_text(text: str, bot) -> Optional[str]:
    return bot.resolve_driver(text, default=None)


# ----------------------------
# Main loop
# ----------------------------
def main() -> None:
    print("F1 Strategy Chatbot (CLI)")
    print("Type 'help' for commands. Type 'exit' to quit.\n")

    state = ChatState()
    ensure_name(state)
    print(
    f"Alright {state.user_name}, what do you want to do?\n"
    "1) Explain a REAL pit stop (e.g. 'why did Verstappen pit on lap 28 in Monaco?')\n"
    "2) Run a WHAT-IF pit simulation (e.g. 'what if Leclerc pits on lap 25 in Abu Dhabi?')\n"
    "3) Get a pit recommendation (e.g. 'should we pit now?')\n"
    "Tip: you can just type naturally — I’ll try to detect driver/race/lap.\n"
    )

    bot: Optional[StrategyService] = None
    last_result: Optional[Dict[str, Any]] = None

    # auto-fill driver if missing and user mentions it
    if bot is not None and state.driver is None:
        d = bot.resolve_driver(user, default=None)
        if d:
            state.driver = d

    while True:
        user = input(f"{state.user_name}> ").strip()
        # Auto-fill context from natural language
        if state.driver is None:
            d = parse_driver_name(user)
            if d:
                state.driver = d
                print(f"✅ Detected driver: {state.driver}")

        if state.race_id is None:
            r = parse_race_alias(user)
            if r:
                state.race_id = r
                print(f"✅ Detected race: {state.race_id}")

        # If user mentions a lap, store it too (optional)
        L = parse_pit_lap(user)
        if L is not None:
            state.lap = L

        if not user:
            continue

        intent = detect_intent(user)

        if intent == "exit":
            print(f"Bye {state.user_name} 👋")
            break

        if intent == "help":
            print_help()
            continue

        if intent == "set":
            parts = user.split(maxsplit=2)
            if len(parts) < 3:
                print("Usage: set race <RaceId> | set driver <CODE> | set lap <N>")
                continue
            key = parts[1].lower()
            value = parts[2].strip()

            if key == "race":
                state.race_id = value
                print(f"✅ race_id set to: {state.race_id}")
            elif key == "driver":
                # Lazy load bot just to resolve names (or do a lightweight resolver in CLI)
                if bot is None:
                    print("Loading strategy engine... (models + data)")
                    from app.services.strategy_service import StrategyService
                    bot = StrategyService()
                    print("✅ Strategy engine loaded.\n")

                resolved = bot.resolve_driver(value, default=value.upper())
                state.driver = resolved
                print(f"✅ driver set to: {state.driver}")
            elif key == "lap":
                try:
                    state.lap = int(value)
                    print(f"✅ lap set to: {state.lap}")
                except ValueError:
                    print("Lap must be an integer.")
            elif key == "name":
                state.user_name = value
                print(f"✅ name set to: {state.user_name}")
            else:
                print("Unknown set key. Use: race, driver, lap, name.")
            continue

        if intent == "status" and user.lower().strip() == "status":
            print("Current state:", asdict(state))
            continue
    
        if intent == "why_real":
            # Ensure we have at least race set; driver can come from message or state
            drv_from_msg, lap_from_msg = parse_why_did_pit(user)
            driver = (drv_from_msg or state.driver)
            if not state.race_id:
                print("Set the race first: set race <RaceId>")
                continue
            if not driver:
                print("Tell me the driver code (e.g., VER/LEC/HAM) or set driver <CODE>.")
                continue

            # load bot if needed
            if bot is None:
                from app.services.strategy_service import StrategyService
                print("Loading strategy engine... (models + data)")
                bot = StrategyService()
                print("✅ Strategy engine loaded.\n")

            pits = bot.get_real_pit_laps(state.race_id, driver)

            if not pits:
                print(f"No pit stops found for {driver} in {state.race_id}.")
                continue

            # If user gave a lap: answer directly
            if lap_from_msg is not None:
                payload = bot.explain_real_pit(state.race_id, driver, pit_lap=lap_from_msg, horizon_laps=20, window=2)
                print(payload.get("summary_viewer", payload["summary"]))
                continue

            # If user did NOT give a lap and there are multiple: ask which
            if len(pits) > 1:
                opts = ", ".join(str(x) for x in pits)
                choice = input(f"{driver} pitted on laps {opts}. Which pit stop do you mean? ").strip()
                try:
                    chosen = int(choice)
                except ValueError:
                    print("Please type a lap number (e.g., 30).")
                    continue

                payload = bot.explain_real_pit(state.race_id, driver, pit_lap=chosen, horizon_laps=20, window=2)
                print(payload.get("summary_viewer", payload["summary"]))
                continue

            # Only one pit stop: use it
            payload = bot.explain_real_pit(state.race_id, driver, pit_lap=pits[0], horizon_laps=20, window=2)
            print(payload.get("summary_viewer", payload["summary"]))
            continue

        missing = require_context(state)
        if missing:
            state.pending_intent = intent
            state.pending_text = user
            # Ask for just the missing bit(s)
            if "race_id" in missing and "driver" in missing:
                print("Which race and driver? (e.g. '2025-Monaco' and 'VER' or 'Verstappen')")
            elif "race_id" in missing:
                print("Which race? (e.g. 'Monaco 2025' or '2025-Monaco')")
            elif "driver" in missing:
                print("Which driver? (e.g. 'VER' or 'Verstappen')")
            continue

        # Lazy-load StrategyService only when needed
        if intent in {"simulate", "recommend", "explain"}:
            if bot is None:
                from app.services.strategy_service import StrategyService
                print("Loading strategy engine... (models + data)")
                bot = StrategyService()
                print("✅ Strategy engine loaded.\n")

        if intent == "simulate":
            pit_lap = parse_pit_lap(user) or state.lap
            if pit_lap is None:
                print("Tell me a lap: e.g. 'what if pit lap 25' or set lap 25")
                continue

            result = bot.simulate(
                race_id=state.race_id,
                driver=state.driver,
                pit_lap=pit_lap,
                horizon_laps=20,
            )
            last_result = {"type": "simulate", "payload": result}
            print(result.get("viewer_summary", result["summary"]))
            continue

        if intent == "recommend":
            if state.lap is None:
                print("Set a current lap first: set lap <N> (needed for recommendations)")
                continue

            candidates = list(range(state.lap, state.lap + 6))  # lap..lap+5
            result = bot.recommend_pit_lap(
                race_id=state.race_id,
                driver=state.driver,
                current_lap=state.lap,
                candidate_laps=candidates,
                horizon_laps=20,
            )
            last_result = {"type": "recommend", "payload": result}
            print(result.get("viewer_summary", result["summary"]))
            continue

        if intent == "explain":
            # Natural language: "why did <driver> pit on lap <n> in <race>?"
            if "why did" in user.lower() and "pit" in user.lower():
                if bot is None:
                    from app.services.strategy_service import StrategyService
                    bot = StrategyService()

                # ensure we have context
                missing = require_context(state)
                if missing:
                    print(f"Missing context: {missing}. Try mentioning race/driver in the question or use set.")
                    continue

                pit_lap = parse_pit_lap(user)
                if pit_lap is None:
                    # if they didn’t say which pit, disambiguate
                    pits = bot.get_real_pit_laps(state.race_id, state.driver)
                    if not pits:
                        print(f"{state.driver} has no pit stops recorded for {state.race_id}.")
                        continue
                    if len(pits) == 1:
                        pit_lap = pits[0]
                    else:
                        print(f"{state.driver} pitted on laps {pits}. Which one do you mean?")
                        continue

                out = bot.explain_real_pit(state.race_id, state.driver, pit_lap=pit_lap, horizon_laps=20, window=2)
                print(out["summary"])
                continue
            if not last_result:
                print("Nothing to explain yet. Run a simulation or recommendation first.")
                continue
            explanation = bot.explain(last_result)
            print(explanation)
            continue
        # If we were waiting for missing context, and we now have it, resume
        if state.pending_intent and not require_context(state):
            user = state.pending_text or user
            intent = state.pending_intent
            state.pending_intent = None
            state.pending_text = None
            # Now continue handling with the resumed intent/text

        print(f"{state.user_name}, I can simulate pit stops, recommend a pit lap, or explain results.")
        print("Try: 'what if pit lap 25' or 'should we pit' or 'why'")
if __name__ == "__main__":
    main()


