#!/usr/bin/env python3
"""Generate synthetic lead management data for ROSHN."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

N_LEADS = 8000
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 12, 31)

# ---- Communities & Zones ----
communities = {
    "North": ["Sedra", "Warefa", "Alarous"],
    "Central": ["Al Nargis", "Al Yasmin", "Al Ahlam"],
    "South": ["Al Bayan", "Marafy"],
    "East": ["Al Jawharah", "Al Rimal"],
    "West": ["Dar Al Salam", "Al Fursan"],
}
all_communities = []
all_zones = []
for zone, comms in communities.items():
    for c in comms:
        all_communities.append(c)
        all_zones.append(zone)

# ---- Lead Sources ----
sources = ["Website", "Social Media", "Referral", "Walk-in", "Exhibition", "Call Center", "Partner Agency", "Email Campaign"]
source_weights = [0.25, 0.20, 0.15, 0.10, 0.10, 0.08, 0.07, 0.05]

# ---- Lead Stages ----
stages = ["New", "Contacted", "Qualified", "Site Visit", "Negotiation", "Proposal Sent", "Won", "Lost"]

# ---- Property Interest ----
property_types = ["Villa", "Townhouse", "Apartment", "Penthouse", "Duplex"]
property_weights = [0.30, 0.25, 0.25, 0.10, 0.10]

# ---- Nationalities ----
nationalities = ["Saudi", "UAE", "Egyptian", "Jordanian", "Indian", "Pakistani", "British", "American", "Filipino", "Other"]
nat_weights = [0.35, 0.10, 0.10, 0.08, 0.10, 0.07, 0.05, 0.05, 0.04, 0.06]

# ---- Budget Ranges (SAR) ----
budget_ranges = ["500K-1M", "1M-2M", "2M-3M", "3M-5M", "5M-10M", "10M+"]
budget_weights = [0.15, 0.25, 0.25, 0.20, 0.10, 0.05]
budget_midpoints = {"500K-1M": 750000, "1M-2M": 1500000, "2M-3M": 2500000, "3M-5M": 4000000, "5M-10M": 7500000, "10M+": 15000000}

# ---- Generate Leads ----
records = []
for i in range(N_LEADS):
    lead_id = f"LEAD-{i+1:06d}"
    
    # Random date
    days_range = (END_DATE - START_DATE).days
    created = START_DATE + timedelta(days=np.random.randint(0, days_range))
    
    # Source
    source = np.random.choice(sources, p=source_weights)
    
    # Community interest
    comm_idx = np.random.randint(0, len(all_communities))
    community = all_communities[comm_idx]
    zone = all_zones[comm_idx]
    
    # Property interest
    prop_type = np.random.choice(property_types, p=property_weights)
    budget = np.random.choice(budget_ranges, p=budget_weights)
    
    # Nationality
    nationality = np.random.choice(nationalities, p=nat_weights)
    
    # Lead scoring (0-100)
    base_score = np.random.normal(50, 20)
    # Boost for referrals and walk-ins
    if source in ["Referral", "Walk-in"]: base_score += 15
    if source == "Exhibition": base_score += 10
    # Boost for higher budgets
    if budget in ["5M-10M", "10M+"]: base_score += 10
    lead_score = int(np.clip(base_score, 5, 100))
    
    # Stage progression based on score and time
    days_since = (END_DATE - created).days
    if lead_score > 75 and days_since > 60:
        stage_probs = [0.02, 0.03, 0.05, 0.10, 0.15, 0.15, 0.35, 0.15]
    elif lead_score > 55 and days_since > 30:
        stage_probs = [0.05, 0.10, 0.15, 0.20, 0.15, 0.15, 0.10, 0.10]
    elif days_since > 14:
        stage_probs = [0.10, 0.20, 0.20, 0.15, 0.10, 0.05, 0.05, 0.15]
    else:
        stage_probs = [0.40, 0.30, 0.15, 0.05, 0.03, 0.02, 0.02, 0.03]
    
    stage = np.random.choice(stages, p=stage_probs)
    
    # Response time (hours)
    if source in ["Walk-in", "Call Center"]:
        response_hours = max(0.1, np.random.exponential(2))
    else:
        response_hours = max(0.5, np.random.exponential(12))
    
    # Interactions count
    stage_idx = stages.index(stage)
    interactions = max(1, int(np.random.normal(stage_idx * 2 + 1, 2)))
    
    # AI-assisted flag
    ai_assisted = np.random.random() < 0.65
    
    # Assigned agent
    agents = ["Agent A", "Agent B", "Agent C", "Agent D", "Agent E", "Agent F"]
    agent = np.random.choice(agents)
    
    # Conversion value (only for Won)
    if stage == "Won":
        conv_value = budget_midpoints[budget] * np.random.uniform(0.85, 1.15)
    else:
        conv_value = 0
    
    # Last activity
    if stage in ["Won", "Lost"]:
        max_days = max(8, min(days_since, 180))
        last_activity = created + timedelta(days=np.random.randint(7, max_days))
    else:
        last_activity = END_DATE - timedelta(days=np.random.randint(0, max(1, min(14, days_since))))
    
    # Follow-up scheduled
    follow_up = stage not in ["Won", "Lost"] and np.random.random() < 0.7
    
    records.append({
        "lead_id": lead_id,
        "created_date": created.strftime("%Y-%m-%d"),
        "source": source,
        "community_interest": community,
        "zone": zone,
        "property_type_interest": prop_type,
        "budget_range": budget,
        "nationality": nationality,
        "lead_score": lead_score,
        "stage": stage,
        "response_time_hours": round(response_hours, 1),
        "total_interactions": interactions,
        "ai_assisted": ai_assisted,
        "assigned_agent": agent,
        "conversion_value_sar": round(conv_value, 0),
        "last_activity_date": last_activity.strftime("%Y-%m-%d"),
        "follow_up_scheduled": follow_up,
    })

df = pd.DataFrame(records)

# Save
out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
df.to_csv(os.path.join(out_dir, "roshn_leads.csv"), index=False)
print(f"Generated {len(df)} leads")
print(f"\nStage distribution:")
print(df["stage"].value_counts().to_string())
print(f"\nSource distribution:")
print(df["source"].value_counts().to_string())
print(f"\nConversion rate: {(df['stage']=='Won').mean()*100:.1f}%")
print(f"AI-assisted: {df['ai_assisted'].mean()*100:.1f}%")
print(f"Total conversion value: {df['conversion_value_sar'].sum()/1e6:.1f}M SAR")
