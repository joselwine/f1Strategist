from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from ML.src.simulator import simulate_pit_rejoin_and_traffic
import re
from difflib import get_close_matches


@dataclass
class SimConfig:
    horizon_laps: int = 20
    pit_loss_s: float = 22.0
    traffic_gap_s: float = 1.5
    traffic_loss_s: float = 0.6
    pass_advantage_s: float = 0.4
    post_pit_push_laps: int = 0
    post_pit_push_boost_s: float = 0.0


class StrategyService:

    def __init__(
        self,
        df_fe: Optional[pd.DataFrame] = None,
        pace_model: Any = None,
        features_pace: Optional[List[str]] = None,
        config: Optional[SimConfig] = None,
    ):
        self.project_root = Path(__file__).resolve().parents[2]

        self.config = config or SimConfig()

        self.df_fe = df_fe if df_fe is not None else self._load_df_fe()
        self.pace_model = pace_model if pace_model is not None else self._load_pace_model()
        self.features_pace = features_pace if features_pace is not None else self._load_features_pace()

        self.feature_rank = self._load_feature_rank_optional()

        self._validate_assets()

        # Store last outputs for explanation
        self._last_sim: Optional[Dict[str, Any]] = None
        self._last_reco: Optional[Dict[str, Any]] = None
        self._driver_alias_map = self._build_driver_alias_map()

    # Loaders
    def _load_df_fe(self) -> pd.DataFrame:
        path = self.project_root / "ML" / "data" / "df_fe.parquet"
        if not path.exists():
            raise FileNotFoundError(f"df_fe not found at {path}")
        return pd.read_parquet(path)
    
    def _load_pace_model(self):
        path = self.project_root / "ML" / "models" / "pace_model.joblib"
        if not path.exists():
            raise FileNotFoundError(f"pace_model not found at {path}")
        return joblib.load(path)

    def _load_features_pace(self) -> List[str]:
        path = self.project_root / "ML" / "models" / "features_pace.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"features_pace.txt not found at {path}. "
            )
        feats = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        return feats


    def _norm(self, s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s\-]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _build_driver_alias_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}

        codes = sorted(self.df_fe["Driver"].dropna().astype(str).unique())
        for c in codes:
            mapping[self._norm(c)] = c.upper()

        name_cols = [c for c in ["DriverName", "FullName", "GivenName", "FamilyName"] if c in self.df_fe.columns]
        if name_cols:
            tmp = self.df_fe[["Driver"] + name_cols].dropna(subset=["Driver"]).copy()
            tmp["Driver"] = tmp["Driver"].astype(str).str.upper()
            for _, r in tmp.iterrows():
                code = r["Driver"]
                for col in name_cols:
                    name = str(r[col])
                    n = self._norm(name)
                    if n:
                        mapping[n] = code
                        # surname-only alias
                        parts = n.split()
                        if len(parts) >= 2:
                            mapping[parts[-1]] = code

                if "GivenName" in tmp.columns and "FamilyName" in tmp.columns:
                    full = self._norm(f"{r['GivenName']} {r['FamilyName']}")
                    if full:
                        mapping[full] = code
                        mapping[full.split()[-1]] = code

        # common driver name aliases 
        manual = {
            "max verstappen": "VER",
            "verstappen": "VER",
            "max": "VER",
            "charles leclerc": "LEC",
            "leclerc": "LEC",
            "lewis hamilton": "HAM",
            "hamilton": "HAM",
            "lando norris": "NOR",
            "norris": "NOR",
            "george russell": "RUS",
            "russell": "RUS",
            "carlos sainz": "SAI",
            "sainz": "SAI",
            "fernando alonso": "ALO",
            "alonso": "ALO",
            "sergio perez": "PER",
            "perez": "PER",
            "oscar piastri": "PIA",
            "piastri": "PIA",
        }
        for k, v in manual.items():
            mapping[self._norm(k)] = v

        return mapping

    def resolve_driver(self, user_text: str, default: str | None = None) -> str | None:
        if not hasattr(self, "_driver_alias_map"):
            self._driver_alias_map = self._build_driver_alias_map()

        t = self._norm(user_text)
        if not t:
            return default

        # 1) exact hit
        if t in self._driver_alias_map:
            return self._driver_alias_map[t]

        # 2) if user typed a sentence, try to find any alias inside it
        hits = []
        for alias, code in self._driver_alias_map.items():
            if alias and alias in t:
                hits.append((len(alias), alias, code))
        if hits:
            hits.sort(reverse=True)
            return hits[0][2]

        # 3) fuzzy match
        aliases = list(self._driver_alias_map.keys())
        close = get_close_matches(t, aliases, n=1, cutoff=0.85)
        if close:
            return self._driver_alias_map[close[0]]

        return default

    def local_feature_impact(self, row_feat: pd.Series, top_k: int = 6) -> List[Dict[str, float]]:
        from ML.src.config import FEATURES_PACE

        # Build a 1-row dataframe
        x0 = pd.DataFrame([row_feat[FEATURES_PACE].to_dict()])
        pred0 = float(self.pace_model.predict(x0)[0])

        impacts = []
        for f in FEATURES_PACE:
            x1 = x0.copy()

            # Baseline: median for numeric, mode for categorical
            if pd.api.types.is_numeric_dtype(self.df_fe[f]):
                base = float(self.df_fe[f].median())
            else:
                base = str(self.df_fe[f].mode().iloc[0])

            x1.loc[0, f] = base
            pred1 = float(self.pace_model.predict(x1)[0])

            impacts.append({
                "feature": f,
                "delta_abs": abs(pred1 - pred0),
                "pred0": pred0,
                "pred1": pred1,
                "actual": x0.loc[0, f],
                "baseline": base,
            })

        impacts.sort(key=lambda d: d["delta_abs"], reverse=True)
        return impacts[:top_k]

    def _validate_assets(self) -> None:
        if self.df_fe is None or self.df_fe.empty:
            raise ValueError("df_fe is empty / not loaded.")

        for col in ["RaceId", "Driver", "LapNumber", "LapTime_s"]:
            if col not in self.df_fe.columns:
                raise ValueError(f"df_fe missing required column: {col}")

        if not hasattr(self.pace_model, "predict"):
            raise ValueError("pace_model must have a .predict() method.")

        missing = [c for c in self.features_pace if c not in self.df_fe.columns]
        if missing:
            raise ValueError(f"df_fe missing pace feature cols: {missing[:10]} ... (total {len(missing)})")

    def _delta_at_lap(self, sim_payload: Dict[str, Any], end_lap: int) -> float:
        df = sim_payload["df"]
        df = df[df["LapNumber"].astype(int) <= int(end_lap)]
        if len(df) == 0:
            return float("nan")
        return float(df["DeltaCum_s"].iloc[-1])

    def _load_feature_rank_optional(self) -> pd.DataFrame | None:
        candidates = [
            self.project_root / "ML" / "models" / "feature_rank_pace.csv",
            self.project_root / "ML" / "models" / "feature_rank.csv",
            self.project_root / "ML" / "models" / "feature_rank.parquet",
        ]
        for p in candidates:
            if p.exists():
                if p.suffix == ".parquet":
                    return pd.read_parquet(p)
                return pd.read_csv(p)
        return None

    def _sim_reasons(self, sim_payload: Dict[str, Any]) -> Dict[str, Any]:
        df = sim_payload["df"]

        total_traffic = float(df["TrafficPenalty_s"].sum()) if "TrafficPenalty_s" in df.columns else 0.0

        worst_row = None
        if "TrafficPenalty_s" in df.columns and len(df):
            worst_idx = int(df["TrafficPenalty_s"].values.argmax())
            worst_row = df.iloc[worst_idx].to_dict()

        rejoin_pos = int(df["WhatIfPos"].iloc[0]) if len(df) else None
        min_gap_ahead = float(df["GapAhead_s"].min()) if ("GapAhead_s" in df.columns and len(df)) else None

        return {
            "rejoin_pos": rejoin_pos,
            "total_traffic_penalty_s": round(total_traffic, 3),
            "worst_traffic_lap": None if worst_row is None else int(worst_row["LapNumber"]),
            "worst_traffic_penalty_s": None if worst_row is None else round(float(worst_row["TrafficPenalty_s"]), 3),
            "min_gap_ahead_s": None if min_gap_ahead is None else round(min_gap_ahead, 3),
        }

    def _feature_context(self, race_id: str, driver: str, pit_lap: int) -> Dict[str, Any]:
        L = int(pit_lap) - 1
        row = self.df_fe[(self.df_fe["RaceId"] == race_id) &
                        (self.df_fe["Driver"] == driver) &
                        (self.df_fe["LapNumber"].astype(int) == L)]
        if row.empty:
            return {"lap_used": L, "note": "No df_fe row found for decision lap."}

        r = row.iloc[0]

        snapshot_keys = [
            "LapNumber", "Position", "LapsSinceLastPit", "TyreLife", "TyreWearFrac",
            "Stint", "StopsSoFar", "IsSC", "IsVSC", "lap_avg_3", "TrackStatus", "Compound"
        ]
        snap = {k: (None if k not in row.columns else r.get(k)) for k in snapshot_keys if k in row.columns}

        # top ranked features
        top_feats = []
        if self.feature_rank is not None and "feature" in self.feature_rank.columns:
            top = self.feature_rank.sort_values(self.feature_rank.columns[-1], ascending=False).head(6)
            top_feats = top["feature"].astype(str).tolist()

        top_vals = {}
        for f in top_feats:
            if f in row.columns:
                v = r.get(f)
                if isinstance(v, (float, np.floating)):
                    v = float(v)
                    v = round(v, 4)
                top_vals[f] = v

        return {
            "lap_used": L,
            "snapshot": snap,
            "top_ranked_features": top_feats,
            "top_feature_values": top_vals,
        }
    
    def _fmt_driver(self, code: str) -> str:
        return code

    def _to_float(self, x, default=None):
        try:
            return float(x)
        except Exception:
            return default
        
    def _order_and_gaps_at_lap(self, race_id: str, lap: int):
        race = self.df_fe[self.df_fe["RaceId"] == race_id].copy()
        race["LapNumber_i"] = race["LapNumber"].astype(int)
        race = race[race["LapNumber_i"] <= int(lap)].copy()

        # cum time up to 'lap' for each driver
        race["LapTime_s"] = pd.to_numeric(race["LapTime_s"], errors="coerce")
        race = race.dropna(subset=["LapTime_s"])
        cum = (
            race.sort_values(["Driver", "LapNumber_i"])
                .groupby("Driver")["LapTime_s"].sum()
                .sort_values()
        )  # Series: index=Driver, value=cum_time

        return cum


    def _neighbors_at_lap(self, race_id: str, driver: str, lap: int):
        snap = self.df_fe[
            (self.df_fe["RaceId"] == race_id) &
            (self.df_fe["LapNumber"].astype(int) == int(lap))
        ].copy()

        if snap.empty or "Position" not in snap.columns:
            return None

        snap["Position"] = pd.to_numeric(snap["Position"], errors="coerce")
        snap = snap.dropna(subset=["Position"]).sort_values("Position")

        me = snap[snap["Driver"] == driver]
        if me.empty:
            return None

        pos = int(me["Position"].iloc[0])

        ahead_row = snap[snap["Position"] == pos - 1]
        behind_row = snap[snap["Position"] == pos + 1]

        ahead = ahead_row["Driver"].iloc[0] if len(ahead_row) else None
        behind = behind_row["Driver"].iloc[0] if len(behind_row) else None

        return {
            "pos": pos,
            "ahead": ahead,
            "behind": behind,
            "gap_ahead_s": None,
            "gap_behind_s": None,
        }
    
    def _has_enough_driver_data(self, race_id: str, driver: str) -> bool:
        sub = self.df_fe[
            (self.df_fe["RaceId"] == race_id) &
            (self.df_fe["Driver"] == driver)
        ].copy()

        if sub.empty:
            return False

        # Requires a sensible number of laps to treat the race as usable
        laps = sub["LapNumber"].dropna().astype(int).unique().tolist() if "LapNumber" in sub.columns else []
        if len(laps) < 10:
            return False

        # Checks whether the driver got near the end of the race
        max_lap_driver = max(laps) if laps else 0
        race_sub = self.df_fe[self.df_fe["RaceId"] == race_id].copy()
        race_laps = race_sub["LapNumber"].dropna().astype(int)
        max_lap_race = int(race_laps.max()) if len(race_laps) else 0

        if max_lap_race == 0:
            return False

        # If driver completed much less than the race distance, treat as non-finisher / unusable
        return max_lap_driver >= max_lap_race - 2
    
    def _uo_lines(self, race_id: str, driver: str, pit_lap: int, k: int = 3):
        snap_lap = max(1, int(pit_lap) - 1)
        nb = self._neighbors_at_lap(race_id, driver, snap_lap)
        if not nb:
            return []

        ahead = nb["ahead"]
        behind = nb["behind"]

        lines = []

        def pits_near(drv):
            pits = self.get_real_pit_laps(race_id, drv)
            return [p for p in pits if int(pit_lap) - k <= int(p) <= int(pit_lap) + k]

        if ahead:
            ahead_pits = pits_near(ahead)
            if any(p < pit_lap for p in ahead_pits):
                p0 = min([p for p in ahead_pits if p < pit_lap])
                lines.append(
                    f"Overcut opportunity: {ahead} was the car ahead and pitted on lap {p0}."
                )
            elif not ahead_pits:
                lines.append(
                    f"Undercut opportunity: {ahead} was the car ahead and had not pitted near lap {pit_lap}."
                )

        if behind:
            behind_pits = pits_near(behind)
            if any(p < pit_lap for p in behind_pits):
                p0 = min([p for p in behind_pits if p < pit_lap])
                lines.append(
                    f"Undercut threat: {behind} was the car behind and pitted on lap {p0}."
                )
            else:
                lines.append(
                    f"Threat check: {behind} was the car behind and had not pitted before lap {pit_lap}."
                )

        return lines

    def _build_viewer_summary_from_sim(self, payload: Dict[str, Any]) -> str:
        race = payload["race_id"]
        drv = payload["driver"]
        lap = payload["pit_lap"]
        dlt = payload["delta_end_s"]

        reasons = payload.get("reasons_sim", {}) or {}
        ctx = payload.get("feature_context", {}) or {}
        snap = (ctx.get("snapshot") or {})

        # key snapshot values 
        wear = self._to_float(snap.get("TyreWearFrac"))
        tyre_age = self._to_float(snap.get("LapsSinceLastPit"))
        compound = snap.get("Compound")
        pos = snap.get("Position")

        rejoin = reasons.get("rejoin_pos")
        traffic = reasons.get("total_traffic_penalty_s", 0.0)

        lines = []
        lines.append(f"{drv} pitted on lap {lap} in {race}.")
        if compound is not None or tyre_age is not None or wear is not None:
            lines.append(
                f"Pre-pit: {compound or 'tyre'} | tyre age {tyre_age:.0f} laps | wear {wear:.2f} | running P{int(pos) if pos is not None else '?'}."
                if wear is not None and tyre_age is not None
                else f"Pre-pit: compound={compound}, position={pos}."
            )

        # translate reasons into English
        if rejoin is not None:
            lines.append(f"Expected rejoin: around P{rejoin}.")
        if traffic is not None:
            lines.append(f"Traffic effect (next {payload['horizon_laps']} laps): ~{float(traffic):.2f}s time loss.")

        lines.append(f"Net effect vs baseline over horizon: {dlt:+.3f}s (negative = faster).")

        return " ".join(lines)

    def _local_sensitivity_rank(
        self,
        snapshot: dict,
        *,
        top_k: int = 5,
    ) -> list[dict]:

        import numpy as np

        # Build a 1-row DataFrame in the exact feature order
        base_row = {f: snapshot.get(f) for f in self.features_pace}
        X0 = pd.DataFrame([base_row])

        # Baseline prediction
        try:
            y0 = float(self.pace_model.predict(X0)[0])
        except Exception:
            return []

        # Define nudges
        nudges = {
            "LapsSinceLastPit": 2,
            "TyreLife": 2,
            "TyreWearFrac": 0.05,
            "LapNumberNorm": 0.02,
            "lap_avg_3": 0.25,
            "Position": 1,
            "StopsSoFar": 1,
            "Stint": 1,
            "IsSC": 1,
            "IsVSC": 1,
        }

        # Categorical alternatives
        cat_alts = {
            "Compound": ["SOFT", "MEDIUM", "HARD"],
            "TrackStatus": ["1", "4", "5", "6"],  
            "Team": None,  
        }

        impacts = []

        for f in self.features_pace:
            if f not in base_row:
                continue
            v = base_row[f]

            # numeric nudge
            if isinstance(v, (int, float, np.integer, np.floating)) and f in nudges:
                X1 = X0.copy()
                X1.loc[0, f] = float(v) + float(nudges[f])
                try:
                    y1 = float(self.pace_model.predict(X1)[0])
                    impacts.append({"feature": f, "delta_pred": y1 - y0, "mode": "numeric"})
                except Exception:
                    pass

            # categorical swap
            if isinstance(v, str) and f in cat_alts and cat_alts[f]:
                for alt in cat_alts[f]:
                    if alt != v:
                        X1 = X0.copy()
                        X1.loc[0, f] = alt
                        try:
                            y1 = float(self.pace_model.predict(X1)[0])
                            impacts.append({"feature": f, "delta_pred": y1 - y0, "mode": "categorical", "alt": alt})
                            break
                        except Exception:
                            continue

        # rank by magnitude
        impacts.sort(key=lambda d: abs(d["delta_pred"]), reverse=True)
        return impacts[:top_k]
    
    def _stop_context_type(self, race_id: str, driver: str, pit_lap: int):
        pits = self.get_real_pit_laps(race_id, driver)
        pits = sorted(pits)

        prev_pit = None
        for p in pits:
            if p < pit_lap:
                prev_pit = p

        snap_lap = max(1, pit_lap - 1)
        snap = self.df_fe[
            (self.df_fe["RaceId"] == race_id) &
            (self.df_fe["Driver"] == driver) &
            (self.df_fe["LapNumber"].astype(int) == snap_lap)
        ]

        compound = None
        if len(snap) and "Compound" in snap.columns:
            compound = str(snap.iloc[0]["Compound"]).upper()

        laps_since_prev_stop = None
        if prev_pit is not None:
            laps_since_prev_stop = pit_lap - prev_pit

        weather_stop = compound in {"INTERMEDIATE", "WET"}
        short_gap_stop = laps_since_prev_stop is not None and laps_since_prev_stop <= 8

        return {
            "weather_stop": weather_stop,
            "short_gap_stop": short_gap_stop,
            "laps_since_prev_stop": laps_since_prev_stop,
            "compound": compound,
        }
    
    def _stop_context_lines(self, stop_ctx: dict) -> list[str]:
        lines = []

        if stop_ctx.get("weather_stop"):
            comp = stop_ctx.get("compound")
            lines.append(
                f"This looks like a weather-related stop, with {comp} tyres involved, so changing track conditions were likely more important than normal tyre wear."
            )

        if stop_ctx.get("short_gap_stop"):
            laps = stop_ctx.get("laps_since_prev_stop")
            lines.append(
                f"This was an unusual short-gap stop, only {laps} laps after the previous stop, which suggests a tactical or race-conditions-driven decision rather than a standard tyre-life stop."
            )

        return lines
    
    def _features_to_natural_language(self, snapshot: dict, ranked: list[dict]) -> list[str]:
        lines = []

        tyre_age = snapshot.get("LapsSinceLastPit")
        comp = snapshot.get("Compound")
        wear = snapshot.get("TyreWearFrac")
        pos = snapshot.get("Position")
        is_sc = snapshot.get("IsSC")
        is_vsc = snapshot.get("IsVSC")

        for r in ranked:
            f = r["feature"]
            d = float(r["delta_pred"])

            if f in {"LapsSinceLastPit", "TyreLife", "TyreWearFrac"}:
                lines.append(
                    f"Tyres were getting old ({comp}, age {tyre_age} laps, wear {wear:.2f}), which the model links to lap-time drop-off."
                )

            elif f == "lap_avg_3":
                lines.append("Recent pace trend suggested performance was starting to drift vs the last few laps.")

            elif f == "Position":
                lines.append(f"Track position mattered (running P{int(pos)}), so the stop was likely about controlling traffic and track position.")

            elif f == "IsSC" or f == "IsVSC":
                if is_sc:
                    lines.append("Safety Car conditions reduce pit-loss, so pitting becomes more attractive.")
                elif is_vsc:
                    lines.append("A Virtual Safety Car can make a pit stop cheaper, which can trigger teams to stop.")

            elif f == "TrackStatus":
                lines.append("Race conditions (track status) made timing more sensitive than usual.")

            # fallback
            else:
                direction = "slower" if d > 0 else "faster"
                lines.append(f"{f} was a key driver in the model (shifted predicted pace {direction}).")

        seen = set()
        out = []
        for l in lines:
            if l not in seen:
                out.append(l)
                seen.add(l)
        return out[:3] 
    
    def _undercut_overcut_signal(self, sims: list[dict], pit_lap: int, eval_after: int = 3) -> dict:
        eval_lap = pit_lap + eval_after

        def get_sim(L):
            for s in sims:
                if s["pit_lap"] == L:
                    return s
            return None

        actual = get_sim(pit_lap)
        earlier = get_sim(pit_lap - 1)
        later = get_sim(pit_lap + 1)

        if not actual:
            return {}

        a = self._delta_at_lap(actual, eval_lap)
        e = self._delta_at_lap(earlier, eval_lap) if earlier else None
        l = self._delta_at_lap(later, eval_lap) if later else None

        out = {"eval_lap": eval_lap, "actual": a, "earlier": e, "later": l}

        label = None
        if e is not None and l is not None:
            if e < a and e <= l:
                label = "undercut"
            elif l < a and l <= e:
                label = "overcut"
            else:
                label = "neutral"
        out["label"] = label
        return out
    
    def _clean_value(self, v):
        import numpy as np
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v

    def _clean_snapshot(self, d: dict) -> dict:
        return {k: self._clean_value(v) for k, v in d.items()}
    
    def _estimate_track_pit_loss(self, race_id: str) -> float:
        race = self.df_fe[self.df_fe["RaceId"] == race_id].copy()
        losses = []

        for drv in race["Driver"].unique():
            pits = self.get_real_pit_laps(race_id, drv)
            for p in pits:
                before = race[(race["Driver"] == drv) &
                            (race["LapNumber"].astype(int).between(p-4, p-2))]
                after = race[(race["Driver"] == drv) &
                            (race["LapNumber"].astype(int).between(p, p+1))]

                if len(before) >= 2 and len(after) >= 1:
                    normal = before["LapTime_s"].median()
                    pit_window = after["LapTime_s"].sum()
                    loss = pit_window - normal * len(after)
                    losses.append(loss)

        return float(np.median(losses)) if losses else 22.0
    
    
    def check_race_integrity(self, race_id: str) -> Dict[str, Any]:
        race = self.df_fe[self.df_fe["RaceId"] == race_id].copy()
        if race.empty:
            return {"ok": False, "error": "race not found"}

        race["LapNumber"] = race["LapNumber"].astype(int)

        drivers = sorted(race["Driver"].unique().tolist())
        max_lap = int(race["LapNumber"].max())
        min_lap = int(race["LapNumber"].min())

        issues = {}
        for d in drivers:
            sub = race[race["Driver"] == d]
            laps = sorted(sub["LapNumber"].unique().tolist())
            if not laps:
                issues[d] = {"type": "no_laps"}
                continue

            # missing lap coverage
            full = set(range(min(laps), max(laps) + 1))
            missing = sorted(full - set(laps))

            # NaN LapTime rows
            nan_lt = int(sub["LapTime_s"].isna().sum()) if "LapTime_s" in sub.columns else None

            if missing or (nan_lt and nan_lt > 0):
                issues[d] = {
                    "missing_laps_count": len(missing),
                    "missing_laps_sample": missing[:10],
                    "nan_laptime_rows": nan_lt,
                    "min_lap": min(laps),
                    "max_lap": max(laps),
                }

        return {
            "ok": True,
            "race_id": race_id,
            "drivers": drivers,
            "min_lap": min_lap,
            "max_lap": max_lap,
            "issues": issues,
            "issues_count": len(issues),
        }

    # Public API
    def simulate(
        self,
        race_id: str,
        driver: str,
        pit_lap: int,
        horizon_laps: Optional[int] = None,
        real_pit_reference_lap : Optional[int] = None,
    ) -> Dict[str, Any]:
        
        if not self._has_enough_driver_data(race_id, driver):
            raise ValueError(f"{driver} did not finish the race, or there is not enough usable data for this race.")
        cfg = self.config
        horizon = horizon_laps if horizon_laps is not None else cfg.horizon_laps

        out_df, quality = simulate_pit_rejoin_and_traffic(
            df=self.df_fe,
            pace_model=self.pace_model,
            race_id=race_id,
            driver_code=driver,
            pit_lap=int(pit_lap),
            horizon_laps=int(horizon),
            pit_loss_s=float(cfg.pit_loss_s),
            pace_features=self.features_pace,
            traffic_gap_s=float(cfg.traffic_gap_s),
            traffic_loss_s=float(cfg.traffic_loss_s),
            pass_advantage_s=float(cfg.pass_advantage_s),
            post_pit_push_laps=int(cfg.post_pit_push_laps),
            post_pit_push_boost_s=float(cfg.post_pit_push_boost_s),
            use_real_post_pit=False,
            verbose=False,
            real_pit_reference_lap=real_pit_reference_lap,
        )

        # Summaries
        delta_end = float(out_df["DeltaCum_s"].iloc[-1]) if len(out_df) else np.nan
        base_pos_end = int(out_df["BasePos"].iloc[-1]) if len(out_df) else -1
        what_pos_end = int(out_df["WhatIfPos"].iloc[-1]) if len(out_df) else -1

        summary = (
            f"[{race_id}] {driver} pit lap {pit_lap} → "
            f"finish pos {what_pos_end} (base {base_pos_end}), "
            f"net Δ = {delta_end:+.3f}s"
        )

        payload = {
            "race_id": race_id,
            "driver": driver,
            "pit_lap": int(pit_lap),
            "horizon_laps": int(horizon),
            "delta_end_s": delta_end,
            "base_pos_end": base_pos_end,
            "whatif_pos_end": what_pos_end,
            "real_pit_reference_lap": int(real_pit_reference_lap) if real_pit_reference_lap is not None else None,
            "df": out_df,  
            "summary": summary,
        }
        
        payload["reasons_sim"] = self._sim_reasons(payload)
        payload["feature_context"] = self._feature_context(race_id, driver, pit_lap)
        payload["quality"] = quality
        reasons = payload["reasons_sim"] or {}  
        viewer_lines = [
            f"What-if: {driver} pits on lap {pit_lap} in {race_id}.",
            f"Time impact over next {horizon} laps: {delta_end:+.3f}s.",
            f"Expected position at end of window: P{what_pos_end} (current baseline P{base_pos_end}).",
        ]
        
        if reasons:
            rejoin_pos = reasons.get("rejoin_pos")
            total_traffic = reasons.get("total_traffic_penalty_s", 0.0)
            worst_lap = reasons.get("worst_penalty_lap")
            worst_penalty = reasons.get("worst_penalty_s")
            min_gap = reasons.get("min_gap_ahead_s")

            if rejoin_pos is not None:
                viewer_lines.append(f"- Likely rejoin position: P{int(rejoin_pos)}.")
            if total_traffic is not None:
                viewer_lines.append(f"- Estimated total traffic cost: {float(total_traffic):.2f}s.")
            if worst_lap is not None and worst_penalty is not None:
                viewer_lines.append(f"- Biggest traffic hit: lap {int(worst_lap)} (+{float(worst_penalty):.2f}s).")
            if min_gap is not None:
                viewer_lines.append(f"- Tightest gap ahead: ~{float(min_gap):.2f}s.")

        payload["summary_viewer"] = "\n".join(viewer_lines)

        detail_lines = [
            f"What-if simulation: [{race_id}] {driver} pits on lap {pit_lap}.",
            f"- Horizon: {horizon} laps.",
            f"- Estimated net effect vs actual strategy: {delta_end:+.3f}s.",
            f"- Expected position: P{what_pos_end} (actual strategy: P{base_pos_end}).",
        ]

        if reasons:
            if reasons.get("rejoin_pos") is not None:
                detail_lines.append(f"- Estimated rejoin position: P{reasons['rejoin_pos']}.")
            if reasons.get("total_traffic_penalty_s") is not None:
                detail_lines.append(
                    f"- Estimated total traffic penalty: {float(reasons['total_traffic_penalty_s']):.3f}s."
                )
            if reasons.get("worst_traffic_lap") is not None:
                detail_lines.append(
                    f"- Biggest traffic impact came on lap {int(reasons['worst_traffic_lap'])}."
                )
            if reasons.get("min_gap_ahead_s") is not None:
                detail_lines.append(
                    f"- Tightest gap ahead was about {float(reasons['min_gap_ahead_s']):.3f}s."
                )

        feature_ctx = payload.get("feature_context") or {}
        snap = feature_ctx.get("snapshot") or {}
        if snap:
            comp = snap.get("Compound")
            age = snap.get("LapsSinceLastPit")
            pos = snap.get("Position")
            if comp is not None or age is not None or pos is not None:
                detail_lines.append(
                    f"- Pre-pit context: compound={comp}, tyre age={age}, position={pos}."
                )

        payload["summary_short"] = payload["summary_viewer"]
        payload["summary_detail"] = "\n".join(detail_lines)

        self._last_sim = payload
        return payload

    def get_real_pit_laps(self, race_id: str, driver: str, debug: bool = False):
        sub = self.df_fe[(self.df_fe["RaceId"] == race_id) & (self.df_fe["Driver"] == driver)].copy()
        if sub.empty:
            return ([], {"error": "no rows for driver/race"}) if debug else []

        pit_col = "is_pit_lap" if "is_pit_lap" in sub.columns else ("IsPitLap" if "IsPitLap" in sub.columns else None)
        if pit_col is None:
            return ([], {"error": "no pit flag column"}) if debug else []

        pits = sorted(set(sub.loc[sub[pit_col].astype(int) == 1, "LapNumber"].astype(int).tolist()))

        if not debug:
            return pits

        # Diagnostics
        laps = sorted(set(sub["LapNumber"].astype(int).tolist()))
        missing_laps = []
        if laps:
            full = set(range(min(laps), max(laps) + 1))
            missing_laps = sorted(full - set(laps))

        # detect pitloss offset 
        pitloss_offset = None
        if pits:
            p = pits[0]
            lt = sub.set_index(sub["LapNumber"].astype(int))["LapTime_s"].astype(float)
            lt_p = float(lt.get(p, float("nan")))
            lt_p1 = float(lt.get(p + 1, float("nan")))
            # if next lap is much slower, treat that as "loss lap"
            if lt_p1 == lt_p1 and lt_p == lt_p and (lt_p1 > lt_p + 8.0):
                pitloss_offset = 1
            else:
                pitloss_offset = 0

        info = {
            "rows": len(sub),
            "min_lap": min(laps) if laps else None,
            "max_lap": max(laps) if laps else None,
            "missing_laps_count": len(missing_laps),
            "missing_laps_sample": missing_laps[:10],
            "pit_col": pit_col,
            "pits": pits,
            "pitloss_offset_guess": pitloss_offset,
        }
        return pits, info

    def _sc_context(self, race_id: str, lap: int, w: int = 2) -> dict:
        sub = self.df_fe[
            (self.df_fe["RaceId"] == race_id) &
            (self.df_fe["LapNumber"].astype(int).between(lap - w, lap + w))
        ].copy()
        if sub.empty:
            return {}

        out = {}
        if "IsSC" in sub.columns:
            out["sc_any"] = bool(int(sub["IsSC"].max()) == 1)
        if "IsVSC" in sub.columns:
            out["vsc_any"] = bool(int(sub["IsVSC"].max()) == 1)
        return out

    def explain_real_pit(
        self,
        race_id: str,
        driver: str,
        pit_lap: Optional[int] = None,
        horizon_laps: Optional[int] = None,
        window: int = 2,
        *,
        prefer: str = "viewer",   
    ) -> Dict[str, Any]:
        
        if not self._has_enough_driver_data(race_id, driver):
            raise ValueError(f"{driver} did not finish the race, or there is not enough usable data for this race.")

        pits = self.get_real_pit_laps(race_id, driver)
        if not pits:
            return {"summary": f"No recorded pit laps found for {driver} in {race_id}."}

        if pit_lap is None:
            pit_lap = pits[0]

        if pit_lap not in pits:
            return {"summary": f"{driver} did not pit on lap {pit_lap} in {race_id}. Real pit laps: {pits}"}

        horizon = int(horizon_laps or self.config.horizon_laps)
        end_lap_fixed = int(pit_lap + horizon)

        # Candidates around the real pit lap
        candidates = list(range(max(1, pit_lap - window), pit_lap + window + 1))

        # Simulate each candidate
        sims = [
            self.simulate(
                race_id=race_id, 
                driver=driver, 
                pit_lap=L, 
                horizon_laps=horizon,
                real_pit_reference_lap=pit_lap,
            )
            for L in candidates
        ]

        sc = self._sc_context(race_id, pit_lap, w=2)        
        for s in sims:
            s["delta_end_fixed_s"] = float(self._delta_at_lap(s, end_lap_fixed))

        sims_sorted = sorted(sims, key=lambda x: x["delta_end_fixed_s"])
        best = sims_sorted[0]
        actual = next(s for s in sims if s["pit_lap"] == pit_lap)

        actual_delta = float(actual["delta_end_fixed_s"])
        for s in sims:
            s["delta_vs_actual_s"] = float(s["delta_end_fixed_s"]) - actual_delta
        
        alternatives = [s for s in sims if int(s["pit_lap"]) != int(pit_lap)]

        best_vs_actual = min(
            alternatives, 
            key=lambda x: float(x.get("delta_vs_actual_s", 0))
        ) if alternatives else actual

        diff_vs_best = float(actual["delta_vs_actual_s"]) - float(best_vs_actual["delta_vs_actual_s"])
        uo = self._undercut_overcut_signal(sims, pit_lap, eval_after=3)

        # Snapshot just before the pit
        snap_lap = max(1, pit_lap - 1)
        uo_lines = self._uo_lines(race_id, driver, pit_lap, k=3)
        stop_ctx = self._stop_context_type(race_id, driver, pit_lap)
        snap = self.df_fe[
            (self.df_fe["RaceId"] == race_id)
            & (self.df_fe["Driver"] == driver)
            & (self.df_fe["LapNumber"].astype(int) == snap_lap)
        ]
        snapshot = snap.iloc[0][self.features_pace].to_dict() if len(snap) else {}
        ctx_lines = self._stop_context_lines(stop_ctx)
        # Clean numpy scalars for printing
        try:
            import numpy as np
            def _clean(v):
                if isinstance(v, (np.floating, np.integer)):
                    return v.item()
                return v
            snapshot = {k: _clean(v) for k, v in snapshot.items()}
        except Exception:
            pass

        earlier = [s for s in sims if int(s["pit_lap"]) < int(pit_lap)]
        later = [s for s in sims if int(s["pit_lap"]) > int(pit_lap)]

        earlier_better = sum(1 for s in earlier if float(s.get("delta_vs_actual_s", 0.0)) < 0)
        later_better = sum(1 for s in later if float(s.get("delta_vs_actual_s", 0.0)) < 0)

        pattern_text = None

        if earlier_better > 0 and later_better == 0:
            pattern_text = "The nearby comparisons suggest an earlier stop may have been slightly stronger than waiting longer."
        if later_better > 0 and earlier_better == 0:
            pattern_text = "The nearby comparisons suggest waiting a little longer may have worked better than stopping earlier."
        if earlier_better > 0 and later_better > 0:
            pattern_text = "The stop window looks finely balanced, with both slightly earlier and slightly later options showing some upside."

        # Reasons from simulator
        avg_pit_loss = self._estimate_track_pit_loss(race_id)
        actual_reasons = actual.get("reasons_sim") or actual.get("reasons") or {}
        best_reasons = best.get("reasons_sim") or best.get("reasons") or {}
        actual_df = actual.get("df")
        ahead_after_stop = None
        gap_ahead_after_stop = None

        if actual_df is not None and len(actual_df):
            try:
                first_row = actual_df.iloc[0]
                ahead_after_stop = first_row.get("AheadDriver")
                gap_ahead_after_stop = first_row.get("GapAhead_s")

                actual_reasons["ahead_driver_after_stop"] = ahead_after_stop
                actual_reasons["gap_ahead_after_stop_s"] = gap_ahead_after_stop
            except Exception:
                pass
        horizon = int(horizon_laps or self.config.horizon_laps)
        end_lap_fixed = int(pit_lap + horizon)
        ranked = self._local_sensitivity_rank(snapshot, top_k=5)
        nl_reasons = self._features_to_natural_language(snapshot, ranked)

        # VIEWER SUMMARY 
        viewer_lines = [
            f"[{driver} pitted on lap {pit_lap} in {race_id}.",
        ]

        pos = snapshot.get("Position") if snapshot else None
        if pos is not None:
            try:
                viewer_lines.append(f"- Running position at the time: P{int(pos)}.")
            except Exception:
                pass

        if stop_ctx.get("weather_stop"):
            viewer_lines.append("- This looks like a weather-driven stop.")
        elif stop_ctx.get("short_gap_stop"):
            laps = stop_ctx.get("laps_since_prev_stop")
            viewer_lines.append(f"- This was an unusual short-gap stop, {laps} laps after the previous stop.")
        else:
            viewer_lines.append("- This looks like a standard strategic stop.")

        if best_vs_actual is not None and int(best_vs_actual["pit_lap"]) != int(pit_lap):
            delta = float(best_vs_actual.get("delta_vs_actual_s", 0.0))
            if delta < 0:
                viewer_lines.append(
                    f"- Best nearby alternative: lap {best_vs_actual['pit_lap']} ({abs(delta):.2f}s better than the actual stop)."
                )
            elif delta > 0:
                viewer_lines.append(
                    f"- Best nearby alternative: lap {best_vs_actual['pit_lap']} ({abs(delta):.2f}s worse than the actual stop)."
                )
            else:
                viewer_lines.append(
                    f"- Best nearby alternative: lap {best_vs_actual['pit_lap']} (effectively equal to the actual stop)."
                )

        if ctx_lines:
            viewer_lines.append("Why this stop type matters:")
            for line in ctx_lines:
                viewer_lines.append(f"- {line}")

        if uo_lines:
            viewer_lines.append("Why the timing mattered (traffic / rivals):")
            for line in uo_lines[:3]:
                viewer_lines.append(f"- {line}")

        if uo and uo.get("label") in {"undercut", "overcut"}:
            if uo["label"] == "undercut":
                viewer_lines.append("Strategic signal: an undercut looked strong here")
            else:
                viewer_lines.append("Strategic signal: an overcut looked strong here")
        
        if uo_lines:
            viewer_lines.append("Why the timing mattered (traffic / rivals):")
            for ln in uo_lines[:3]:
                viewer_lines.append(f"- {ln}")
        
        if pattern_text:
            viewer_lines.append(pattern_text)

        if actual_reasons:
            viewer_lines.append(
                f"Traffic/rejoin: rejoin ~P{actual_reasons.get('rejoin_pos')}, "
                f"traffic cost ~{actual_reasons.get('total_traffic_penalty_s', 0.0):.3f}s."
            )

        if nl_reasons:
            viewer_lines.append("Why this stop made sense:")
            for line in nl_reasons[:3]:
                viewer_lines.append(f"- {line}")

        if snapshot:
            comp = snapshot.get("Compound")
            age = snapshot.get("LapsSinceLastPit")
            wear = snapshot.get("TyreWearFrac")
            pos = snapshot.get("Position")
            if pos is not None:
                viewer_lines.append(f"At the time of the stop, {driver} was running P{int(pos)}.")
            viewer_lines.append(
                f"Before the stop (lap {snap_lap}): {comp} tyres, age {age} laps, wear {float(wear):.2f} (0–1), running P{int(pos)}."
            )

        short_lines = [
            f"{driver} pitted on lap {pit_lap} in {race_id}."
        ]

        pos = snapshot.get("Position") if snapshot else None
        if pos is not None:
            try:
                short_lines.append(f"Race position at stop: P{int(pos)}.")
            except Exception:
                pass

        if stop_ctx.get("weather_stop"):
            short_lines.append("This appears to be a weather-driven stop.")
        elif stop_ctx.get("short_gap_stop"):
            laps = stop_ctx.get("laps_since_prev_stop")
            short_lines.append(f"This was an unusual short-gap stop, only {laps} laps after the previous stop.")
        else:
            short_lines.append("This appears to be a standard strategic stop.")

        rejoin_pos = actual_reasons.get("rejoin_pos")
        ahead_driver = actual_reasons.get("ahead_driver_after_stop")
        gap_ahead = actual_reasons.get("gap_ahead_after_stop_s")

        if rejoin_pos is not None:
            short_lines.append(f"Projected rejoin position: P{int(rejoin_pos)}.")

        if ahead_driver and gap_ahead is not None:
            short_lines.append(f"Car ahead after stop: {ahead_driver} (~{float(gap_ahead):.2f}s ahead).")
        elif gap_ahead is not None:
            short_lines.append(f"Gap to car ahead after stop: ~{float(gap_ahead):.2f}s.")

        if best_vs_actual is not None and int(best_vs_actual["pit_lap"]) != int(pit_lap):
            delta = float(best_vs_actual.get("delta_vs_actual_s", 0.0))
            if delta < 0:
                short_lines.append(
                    f"Best nearby alternative: lap {best_vs_actual['pit_lap']} (~{abs(delta):.1f}s faster)."
                )
            elif delta > 0:
                short_lines.append("The actual stop performed better than the nearby alternatives checked.")

        if pattern_text:
            short_lines.append(pattern_text)

        if uo_lines:
            short_lines.append(uo_lines[0])

        summary_short = "\n".join(short_lines)
        summary_viewer = "\n".join(viewer_lines)


        # Detail summary
        detail_lines = [
            f"Real pit explanation: [{race_id}] {driver} pitted on lap {pit_lap}.",
            f"- Compared laps: {candidates[0]} to {candidates[-1]} (a ±{window} lap window).",
            f"- All options are compared at lap {end_lap_fixed}, which is {horizon} laps after the actual stop.",
            f"- The best nearby alternative option was lap {best['pit_lap']}",
            f"- The actual stop was {abs(float(diff_vs_best)):.3f}s {'worse' if float(diff_vs_best) > 0 else 'better'} than the best nearby option.",
        ]

        if actual_reasons:
            detail_lines += [
                f"- Actual rejoin position: P{actual_reasons.get('rejoin_pos')}",
                f"- Actual total traffic penalty: {actual_reasons.get('total_traffic_penalty_s', 0.0):.3f}s",
                f"- Tightest gap ahead (horizon): {actual_reasons.get('min_gap_ahead_s', float('nan'))}",
            ]

        if snapshot:
            detail_lines.append(f"- Snapshot (lap {snap_lap}) features: {snapshot}")

        summary_detail = "\n".join(detail_lines)

        summary = summary_viewer if prefer == "viewer" else summary_detail

        return {
            "race_id": race_id,
            "driver": driver,
            "real_pit_lap": pit_lap,
            "real_pit_laps": pits,
            "window": window,
            "horizon_laps": horizon,
            "end_lap_fixed": end_lap_fixed,
            "actual": actual,
            "best_in_window": best,
            "counterfactuals": [
                {
                    "pit_lap": s["pit_lap"],
                    "delta_vs_actual_s": float(s.get("delta_vs_actual_s", 0.0)),
                    "whatif_pos_end_s": s.get("whatif_pos_end"),
                    "label": (
                        "Actual stop"
                        if int(s["pit_lap"]) == int(pit_lap)
                        else ("Better than actual" if float(s.get("delta_vs_actual_s", 0.0)) < 0 else "Worse than actual")
                    )
                }
                for s in sorted(sims, key=lambda x: float(x.get("delta_vs_actual_s", 0.0)))
            ],
            "stop_context": stop_ctx,
            "snapshot": snapshot,
            "summary_short": summary_short,
            "summary_viewer": summary_viewer,
            "summary_detail": summary_detail,
            "summary": summary,  
        }

    def _reasons_to_viewer_text(self, reasons: Dict[str, Any]) -> str:
        if not reasons:
            return ""

        rejoin = reasons.get("rejoin_pos")
        traffic = reasons.get("total_traffic_penalty_s")
        worst_lap = reasons.get("worst_traffic_lap")
        worst_pen = reasons.get("worst_traffic_penalty_s")
        min_gap = reasons.get("min_gap_ahead_s")

        bits = []
        if rejoin is not None:
            bits.append(f"Rejoin looked like ~P{int(rejoin)}")
        if traffic is not None:
            bits.append(f"traffic cost ≈ {float(traffic):.2f}s over the horizon")
        if worst_lap is not None and worst_pen is not None:
            bits.append(f"biggest traffic hit was lap {int(worst_lap)} (+{float(worst_pen):.2f}s)")
        if min_gap is not None and min_gap == min_gap:  # not NaN
            bits.append(f"tightest gap ahead was ~{float(min_gap):.2f}s")

        return " • ".join(bits)
    
    def recommend_pit_lap(
        self,
        race_id: str,
        driver: str,
        current_lap: int,
        candidate_laps: List[int],
        horizon_laps: Optional[int] = None,
    ) -> Dict[str, Any]:
        
        if not self._has_enough_driver_data(race_id, driver):
            raise ValueError(f"{driver} did not finish the race, or there is not enough usable data for this race.")

        sims = []
        for L in candidate_laps:
            if L < current_lap:
                continue
            sims.append(self.simulate(race_id=race_id, driver=driver, pit_lap=int(L), horizon_laps=horizon_laps))

        if not sims:
            return {"summary": "No valid candidate laps to evaluate.", "summary_viewer": "No valid candidate laps to evaluate."}

        sims_sorted = sorted(sims, key=lambda x: x["delta_end_s"])
        best = sims_sorted[0]
    
        best_delta = float(best["delta_end_s"])
        best_pos = best.get("whatif_pos_end")

        table = []
        for s in sims_sorted:
            delta_vs_best = float(s["delta_end_s"]) - best_delta
            pos_end = s.get("whatif_pos_end")

            if int(s["pit_lap"]) == int(best["pit_lap"]):
                label = "Recommended"
            else:
                time_label = "slower" if delta_vs_best > 0 else "equal"

                if best_pos is not None and pos_end is not None:
                    if int(pos_end) < int(best_pos):
                        pos_label = "better position"
                    elif int(pos_end) > int(best_pos):
                        pos_label = "worse position"
                    else:
                        pos_label = "same position"
                else:
                    pos_label = "position unclear"

                label = f"{time_label}, {pos_label}"

            table.append({
                "pit_lap": int(s["pit_lap"]),
                "delta_vs_best_s": round(delta_vs_best, 3),
                "whatif_pos_end": pos_end,
                "label": label,
            })

        summary = (
            f"Recommendation for {driver} @ {race_id} (current lap {current_lap}):\n"
            f"Best pit lap: {best['pit_lap']} → Δ={best['delta_end_s']:+.3f}s, "
            f"finish pos {best['whatif_pos_end']}\n"
            f"Checked: {', '.join(str(x['pit_lap']) for x in table)}"
        )
        second = sims_sorted[1] if len(sims_sorted) > 1 else None
        diff_vs_2nd = None
        short_lines = [
            f"Recommended pit lap: {best['pit_lap']} for {driver} in {race_id}.",
            "This was the strongest option in the tested window.",
        ]

        if diff_vs_2nd is not None:
            short_lines.append(
                f"It was about {abs(diff_vs_2nd):.3f}s better than the next-best alternative."
            )

        if best.get("whatif_pos_end") is not None:
            short_lines.append(f"Expected position after the comparison window: P{best['whatif_pos_end']}.")
        if second is not None:
            diff_vs_2nd = float(best["delta_end_s"]) - float(second["delta_end_s"])

        detail_lines = [
            f"Recommendation analysis for {driver} in {race_id}.",
            f"- Current lap: {current_lap}.",
            f"- Candidate laps tested: {', '.join(str(x['pit_lap']) for x in table)}.",
            f"- Recommended pit lap: {best['pit_lap']}.",
            f"- Estimated effect over the next {horizon_laps or self.config.horizon_laps} laps: {float(best['delta_end_s']):+.3f}s vs actual strategy.",
            f"- Expected position: P{best.get('whatif_pos_end', '?')}.",
        ]

        if diff_vs_2nd is not None:
            detail_lines.append(f"- Margin to next-best option: {diff_vs_2nd:+.3f}s.")

        reasons = best.get("reasons_sim") or best.get("reasons") or {}
        if reasons:
            if reasons.get("rejoin_pos") is not None:
                detail_lines.append(f"- Estimated rejoin position: P{reasons['rejoin_pos']}.")
            if reasons.get("total_traffic_penalty_s") is not None:
                detail_lines.append(
                    f"- Estimated traffic penalty: {float(reasons['total_traffic_penalty_s']):.3f}s."
                )
            if reasons.get("worst_traffic_lap") is not None:
                detail_lines.append(
                    f"- Biggest traffic impact came on lap {int(reasons['worst_traffic_lap'])}."
                )

        viewer_lines = [
            f"Recommended pit lap: {best['pit_lap']} for {driver} in {race_id}.",
            f"- Estimated time impact over the comparison window: {best['delta_end_s']:+.3f}s.",
            f"- Expected position: P{best['whatif_pos_end']} (baseline P{best['base_pos_end']}).",
        ]

        if diff_vs_2nd is not None and second is not None:
            if diff_vs_2nd < 0:
                viewer_lines.append(
                    f"- Margin to next-best option: {abs(diff_vs_2nd):.2f}s better than lap {second['pit_lap']}."
                )
            elif diff_vs_2nd > 0:
                viewer_lines.append(
                    f"- Margin to next-best option: {abs(diff_vs_2nd):.2f}s worse than lap {second['pit_lap']}."
                )
            else:
                viewer_lines.append(
                    f"- Margin to next-best option: effectively equal to lap {second['pit_lap']}."
                )

        reasons = best.get("reasons_sim") or {}
        if reasons:
            rejoin_pos = reasons.get("rejoin_pos")
            total_traffic = reasons.get("total_traffic_penalty_s", 0.0)
            min_gap = reasons.get("min_gap_ahead_s")

            if rejoin_pos is not None:
                viewer_lines.append(f"- Likely rejoin position: P{int(rejoin_pos)}.")
            if total_traffic is not None:
                viewer_lines.append(f"- Estimated traffic cost: {float(total_traffic):.2f}s.")
            if min_gap is not None:
                viewer_lines.append(f"- Tightest gap ahead: ~{float(min_gap):.2f}s.")

        payload = {
            "race_id": race_id,
            "driver": driver,
            "current_lap": int(current_lap),
            "candidates": candidate_laps,
            "best": best,
            "second_best": second,
            "counterfactuals": table,
            "summary": summary,
            "summary_short": "\n".join(short_lines),
            "summary_viewer": "\n".join(viewer_lines),
            "summary_detail": "\n".join(detail_lines),
        }

        self._last_reco = payload
        return payload

    def explain(self, last_result: Dict[str, Any]) -> str:
        rtype = last_result.get("type")
        payload = last_result.get("payload", {})

        if rtype == "simulate":
            df = payload["df"]
            total_pen = float(df["TrafficPenalty_s"].sum()) if "TrafficPenalty_s" in df.columns else 0.0
            delta_end = payload["delta_end_s"]
            pit_lap = payload["pit_lap"]

            return (
                f"Explanation for pit lap {pit_lap}:\n"
                f"- Net delta at horizon end: {delta_end:+.3f}s\n"
                f"- Total traffic penalty applied: {total_pen:.3f}s\n"
                f"- Biggest swings usually come from: pit-loss lap, traffic lock-ups, and pace delta.\n"
                f"(Next: we’ll attach feature-importance to explain the predicted pace deltas.)"
            )

        if rtype == "recommend":
            best = payload.get("best", {})
            second = payload.get("second_best")

            lines = []
            lines.append("Why this recommendation:")

            # Core objective
            lines.append(f"- Objective: minimise Δ at horizon end (lower is better).")
            lines.append(f"- Best: lap {best.get('pit_lap')} → Δ={best.get('delta_end_s'):+.3f}s, finish pos {best.get('whatif_pos_end')}")

            # Sim reasons
            rs = best.get("reasons_sim", {})
            if rs:
                lines.append(f"- Rejoin position: P{rs.get('rejoin_pos')}, total traffic penalty: {rs.get('total_traffic_penalty_s')}s")
                if rs.get("worst_traffic_lap") is not None:
                    lines.append(f"- Worst traffic lap: {rs['worst_traffic_lap']} (+{rs['worst_traffic_penalty_s']}s)")
                if rs.get("min_gap_ahead_s") is not None:
                    lines.append(f"- Tightest gap ahead in horizon: {rs['min_gap_ahead_s']}s")

            # Compare to 2nd best
            if second is not None:
                diff = float(best["delta_end_s"]) - float(second["delta_end_s"])
                lines.append(f"- vs 2nd-best (lap {second['pit_lap']}): best is {diff:+.3f}s different (negative means better).")

            # Feature context
            fc = best.get("feature_context", {})
            if fc and "top_ranked_features" in fc:
                lines.append(f"- Top ranked pace drivers: {', '.join(fc['top_ranked_features'][:5])}")
                if fc.get("top_feature_values"):
                    kv = ", ".join([f"{k}={v}" for k, v in list(fc["top_feature_values"].items())[:5]])
                    lines.append(f"- Your values at lap {fc.get('lap_used')}: {kv}")

            return "\n".join(lines)
        return "Nothing to explain yet."
