# Schema Reference

The world model schema decomposes civilization into **19 layers and 857 fields**. Each layer captures a different domain — from sub-minute market ticks to multi-decade demographic shifts. This page visualizes every layer with its internal structure and the reasoning behind each decomposition.

---

## Layer relationships

```mermaid
graph TD
    subgraph "Planetary (slow, structural)"
        PHY[Physical]
        RES[Resources]
        TECH[Technology]
        BIO[Biology]
    end

    subgraph "Infrastructure"
        INF[Infrastructure]
        CYB[Cyber]
        SPC[Space]
    end

    subgraph "Human Systems"
        HLT[Health]
        EDU[Education]
        DEM[Demographics]
    end

    subgraph "Governance"
        POL[Political]
        LEG[Legal]
        INT[Interventions]
    end

    subgraph "Markets & Economy"
        FIN[Financial]
        MAC[Macro]
        SEC[Sector]
        SUP[Supply Chain]
    end

    subgraph "Actors & Beliefs"
        BUS[Business]
        IND[Individual]
        NAR[Narratives]
    end

    subgraph "Meta & Output"
        EVT[Events]
        TRU[Trust]
        REG[Regime]
        FOR[Forecasts]
    end

    PHY --> RES
    RES --> FIN
    MAC --> FIN
    POL --> INT
    INT --> MAC
    NAR --> FIN
    BUS --> SEC
    SEC --> MAC
    REG --> FOR
    DEM --> MAC
    TECH --> SEC
    BIO --> HLT

    style REG fill:#ff6,stroke:#333,stroke-width:3px
    style FOR fill:#6f6,stroke:#333,stroke-width:3px
```

### Entity connection topology

The block diagram above shows layer-to-layer causal arrows. The bubble diagrams below show how **entity instances** connect through **coarse-grained (CG) bridge nodes**. When the schema is compiled onto the canvas, `compile_schema` creates a CG aggregate at each nesting level — a single canvas position that summarizes all fields below it. CG nodes (hexagons below) serve as information bottlenecks, enabling efficient cross-entity and cross-layer attention without O(n^2) all-to-all connections.

#### Full world instance map

All 25 top-level `World` fields. Dense always-on layers appear as circles. Entity instances appear as hexagons — each hexagon is that entity's coarse-grained bridge node, summarizing its entire internal state into a single canvas position that connects to the rest of the world.

```mermaid
graph TD
    subgraph structural["Structural Substrate"]
        PHY((Physical))
        RES((Resources))
        TECH((Technology))
        BIO((Biology))
    end

    subgraph infragroup["Infrastructure & Context"]
        INFRA((Infrastructure))
        CYB((Cyber))
        SPC((Space))
        HLT((Health))
        EDU((Education))
        LEG((Legal))
    end

    subgraph marketgroup["Markets & Beliefs"]
        FIN((Financial))
        NAR((Narratives))
    end

    subgraph countrygroup["Countries"]
        US{{country_us}}
        CN{{country_cn}}
        EU{{country_eu}}
    end

    subgraph sectorgroup["Sectors"]
        ST{{sector_tech}}
        SE{{sector_energy}}
        SF{{sector_fin}}
    end

    subgraph scgroup["Supply Chain Nodes"]
        SCS{{sc_semi}}
        SCE{{sc_energy}}
        SCF{{sc_food}}
    end

    subgraph firmgroup["Firms"]
        FA{{firm_alpha}}
        FB{{firm_beta}}
    end

    subgraph persongroup["Individuals"]
        PA{{person_alpha}}
        PB{{person_beta}}
    end

    subgraph metagroup["Meta & Output"]
        EVT((Events))
        TRU((Trust))
        REG((Regime))
        INT((Interventions))
        FOR((Forecasts))
    end

    PHY --> RES
    RES --> FIN
    TECH --> FIN
    BIO --> HLT

    PA -->|org_link| FA
    PB -->|org_link| FB
    FA -->|sector_link| ST
    FB -->|sector_link| SE
    FA -->|geo_link| US
    FB -->|geo_link| CN
    SCS --> ST
    SCE --> SE
    SCF --> RES

    US --> FIN
    US --> REG
    CN --> FIN
    EU --> FIN
    ST --> TECH
    SE --> RES
    SF --> FIN
    FA --> EVT
    FB --> EVT

    REG --> FOR
    EVT --> NAR
    NAR --> FIN
    INT --> FOR

    style US fill:#4a9,stroke:#333,color:#fff
    style CN fill:#4a9,stroke:#333,color:#fff
    style EU fill:#4a9,stroke:#333,color:#fff
    style ST fill:#c84,stroke:#333,color:#fff
    style SE fill:#c84,stroke:#333,color:#fff
    style SF fill:#c84,stroke:#333,color:#fff
    style FA fill:#48c,stroke:#333,color:#fff
    style FB fill:#48c,stroke:#333,color:#fff
    style PA fill:#84c,stroke:#333,color:#fff
    style PB fill:#84c,stroke:#333,color:#fff
    style SCS fill:#aa4,stroke:#333
    style SCE fill:#aa4,stroke:#333
    style SCF fill:#aa4,stroke:#333
    style REG fill:#ff6,stroke:#333,stroke-width:3px
    style FOR fill:#6f6,stroke:#333,stroke-width:3px
```

#### Cross-entity bridge chain

Entities form a hierarchy through explicit link fields (`org_link`, `sector_link`, `geography_link`). Each entity type's internal sub-components (circles) flow into that entity's CG bridge node (hexagon). Thick arrows show inter-entity bridges; dotted arrows show outward connections to dense layers.

```mermaid
graph LR
    subgraph ind["Individual"]
        I_COG((Cognitive))
        I_INC((Incentives))
        I_NET((Network))
        I_STA((State))
        I_CG{{Individual CG}}
        I_COG --> I_CG
        I_INC --> I_CG
        I_NET --> I_CG
        I_STA --> I_CG
    end

    subgraph biz["Business"]
        B_FIN((Financials))
        B_OPS((Operations))
        B_STR((Strategy))
        B_MKT((Market Pos))
        B_RSK((Risk))
        B_SC((SupplyChain))
        B_CG{{Business CG}}
        B_FIN --> B_CG
        B_OPS --> B_CG
        B_STR --> B_CG
        B_MKT --> B_CG
        B_RSK --> B_CG
        B_SC --> B_CG
    end

    subgraph sec["Sector"]
        S_DEM((Demand))
        S_SUP((Supply))
        S_PRO((Profitability))
        S_STR((Structural))
        S_CG{{Sector CG}}
        S_DEM --> S_CG
        S_SUP --> S_CG
        S_PRO --> S_CG
        S_STR --> S_CG
    end

    subgraph cty["Country"]
        C_MAC((Macro))
        C_POL((Political))
        C_DEM((Demographics))
        C_CG{{Country CG}}
        C_MAC --> C_CG
        C_POL --> C_CG
        C_DEM --> C_CG
    end

    I_CG ==>|org_link| B_CG
    B_CG ==>|sector_link| S_CG
    B_CG ==>|geography_link| C_CG

    C_CG -.-> DL((Dense Layers))
    S_CG -.-> DL
    B_CG -.-> DL
    I_CG -.-> DL

    style I_CG fill:#84c,stroke:#333,color:#fff
    style B_CG fill:#48c,stroke:#333,color:#fff
    style S_CG fill:#c84,stroke:#333,color:#fff
    style C_CG fill:#4a9,stroke:#333,color:#fff
    style DL fill:#eee,stroke:#999,stroke-dasharray:5
```

#### Country coarse-graining detail

A Country entity contains three major sub-schemas, each with its own CG bridge. Sub-component CG nodes (inner hexagons) aggregate their fields, then flow into the top-level Country CG (outer hexagon) that bridges to dense layers and other entities.

```mermaid
graph LR
    subgraph macro["MacroEconomy"]
        OG((Output &<br/>Growth))
        IS((Inflation))
        LM((Labor<br/>Market))
        FS((Fiscal))
        TB((Trade))
        HM((Housing))
        MAC_CG{{Macro CG}}
        OG --> MAC_CG
        IS --> MAC_CG
        LM --> MAC_CG
        FS --> MAC_CG
        TB --> MAC_CG
        HM --> MAC_CG
    end

    subgraph pol["PoliticalLayer"]
        EX((Executive))
        LE((Legislative))
        JU((Judicial))
        GEO((Geopolitical))
        IQ((Institutional<br/>Quality))
        POL_CG{{Political CG}}
        EX --> POL_CG
        LE --> POL_CG
        JU --> POL_CG
        GEO --> POL_CG
        IQ --> POL_CG
    end

    subgraph dem["DemographicLayer"]
        DP((Population, Dependency,<br/>Urbanization, Fertility, ...))
        DEM_CG{{Demographics CG}}
        DP --> DEM_CG
    end

    CTR_CG{{"Country CG"}}
    MAC_CG ==> CTR_CG
    POL_CG ==> CTR_CG
    DEM_CG ==> CTR_CG

    CTR_CG --> FIN((Financial))
    CTR_CG --> NAR((Narratives))
    CTR_CG --> REG((Regime))
    CTR_CG --> FOR((Forecasts))
    CTR_CG --> LEG((Legal))
    CTR_CG -.-> SEC{{Sectors}}
    CTR_CG -.-> BIZ{{Firms}}

    style MAC_CG fill:#4a9,stroke:#333,color:#fff
    style POL_CG fill:#4a9,stroke:#333,color:#fff
    style DEM_CG fill:#4a9,stroke:#333,color:#fff
    style CTR_CG fill:#2d7,stroke:#333,color:#fff,stroke-width:3px
```

#### Business coarse-graining detail

A Business entity has six sub-schemas (including an embedded SupplyChainNode). Each flows through the Business CG bridge, which connects outward via `sector_link` and `geography_link` to Sector and Country entities, and directly to Financial, Events, and Forecasts layers.

```mermaid
graph LR
    subgraph biz["Business Internal"]
        FF((Financials<br/>18 fields))
        FO((Operations<br/>10 fields))
        FS2((Strategy<br/>10 fields))
        FM((Market Pos<br/>8 fields))
        FR((Risk<br/>8 fields))
        SC((SupplyChain<br/>9 fields))
        BIZ_CG{{Business CG}}
        FF --> BIZ_CG
        FO --> BIZ_CG
        FS2 --> BIZ_CG
        FM --> BIZ_CG
        FR --> BIZ_CG
        SC --> BIZ_CG
    end

    BIZ_CG -->|sector_link| SEC{{Sector CG}}
    BIZ_CG -->|geography_link| CTY{{Country CG}}
    BIZ_CG --> FIN((Financial))
    BIZ_CG --> EVT((Events))
    BIZ_CG --> FOR((Forecasts))
    BIZ_CG -.->|org_link| IND{{Individual CG}}

    style BIZ_CG fill:#48c,stroke:#333,color:#fff,stroke-width:3px
    style SEC fill:#c84,stroke:#333,color:#fff
    style CTY fill:#4a9,stroke:#333,color:#fff
    style IND fill:#84c,stroke:#333,color:#fff
```

---

## 1. Planetary Physical Layer

**Frequency**: τ6–τ7 (annual to multi-year) · **Fields**: 17

Climate, geography, and natural disasters — the slowest structural forces that constrain everything above them. An earthquake disrupts supply chains. A shifting monsoon pattern moves food prices. These are boundary conditions on civilization.

```mermaid
graph TD
    subgraph PlanetaryPhysicalLayer
        subgraph ClimateState
            A1[global_temp_anomaly<br/>τ6 annual]
            A2[enso_phase<br/>τ5 quarterly]
            A3[monsoon_state<br/>τ5 quarterly]
            A4[polar_vortex_stability<br/>τ5 quarterly]
            A5[extreme_weather_freq<br/>τ4 monthly]
            A6[sea_level_trend<br/>τ7 decadal]
            A7[carbon_ppm<br/>τ6 annual]
        end
        subgraph GeographicInfrastructure
            B1[shipping_lane_capacity<br/>τ6 annual]
            B2[chokepoint_risk<br/>τ4 monthly]
            B3[undersea_cable_topology<br/>τ6 annual]
            B4[rail_freight_network<br/>τ6 annual]
            B5[port_congestion<br/>τ2 daily]
            B6[air_freight_utilization<br/>τ2 daily]
        end
        subgraph DisasterLayer
            C1[seismic_risk_structural<br/>τ6 annual]
            C2[active_disaster_state<br/>τ0 tick]
            C3[pandemic_risk<br/>τ3 weekly]
            C4[volcanic_risk<br/>τ5 quarterly]
            C5[wildfire_state<br/>τ2 daily]
        end
    end
```

**Why this decomposition?** Climate operates on its own timescale but constrains resources (crop yields, energy demand). Geographic infrastructure — shipping lanes, cables, rail networks — changes slowly but creates chokepoints that matter in crises. Disasters are rare high-impact events that propagate through every other layer.

---

## 2. Resource Layer

**Frequency**: τ1–τ4 (hourly to monthly) · **Fields**: 45

Physical inputs to production: energy, metals, food, water, and compute. These are the atoms of the economy — priced in real-time, constrained by geology, and shaped by geopolitics.

```mermaid
graph TD
    subgraph ResourceLayer
        subgraph EnergySystem
            E1[crude_price τ0]
            E2[crude_inventory τ3]
            E3[crude_production_capacity τ4]
            E4[natgas_price τ0]
            E5[natgas_storage τ3]
            E6[lng_shipping_rates τ2]
            E7[coal_price τ2]
            E8[electricity_grid_load τ1]
            E9[renewable_generation τ1]
            E10[refinery_utilization τ3]
            E11[strategic_reserves τ4]
            E12[opec_spare_capacity τ4]
            E13[energy_transition_pace τ6]
        end
        subgraph MetalsAndMinerals
            M1[copper τ0]
            M2[iron_ore τ2]
            M3[lithium τ3]
            M4[rare_earths τ4]
            M5[aluminum τ2]
            M6[nickel τ2]
            M7[gold τ0]
            M8[silver τ0]
            M9[mining_capex_cycle τ5]
        end
        subgraph FoodSystem
            F1[wheat τ2]
            F2[corn τ2]
            F3[soybean τ2]
            F4[rice τ3]
            F5[fertilizer_prices τ3]
            F6[food_price_index τ4]
            F7[crop_yield_forecast τ5]
            F8[food_insecurity τ4]
            F9[arable_land_trend τ6]
        end
        subgraph WaterStress
            W1[aquifer_depletion τ6]
            W2[drought_index τ4]
            W3[desalination_capacity τ6]
            W4[water_conflict_risk τ5]
        end
        subgraph ComputeSupply
            CS1[gpu_spot_price τ1]
            CS2[datacenter_capacity τ4]
            CS3[fab_utilization τ4]
            CS4[leading_edge_capacity τ5]
            CS5[chip_inventory_days τ3]
            CS6[ai_training_demand τ3]
            CS7[semiconductor_capex τ5]
            CS8[export_control_severity τ4]
        end
    end
```

**Why this decomposition?** Energy prices (tick-level) sit atop slower inventory and capacity cycles (weekly/monthly). Metals and food are priced separately but share supply chain dependencies. Water stress is a slow-burn structural risk that shows up in food and energy costs. Compute supply is the 21st-century equivalent of electricity — a resource with its own pricing, capacity constraints, and geopolitical dimensions (export controls).

---

## 3. Global Financial Layer

**Frequency**: τ0–τ2 (sub-minute to daily) · **Fields**: 68

The fastest-moving layer. Yield curves, credit spreads, FX, equities, liquidity, and crypto — all interconnected through arbitrage and reflexivity.

```mermaid
graph TD
    subgraph GlobalFinancialLayer
        subgraph CentralBankState
            CB1[policy_rate τ4]
            CB2[balance_sheet_size τ3]
            CB3[forward_guidance_hawkish τ4]
            CB4[qe_qt_pace τ4]
            CB5["credibility τ5 (w=2.0)"]
            CB6[dot_plot_median τ5]
        end
        subgraph YieldCurveState
            Y1[short_rate τ0]
            Y2[two_year τ0]
            Y3[five_year τ0]
            Y4[ten_year τ0]
            Y5[thirty_year τ0]
            Y6[term_premium τ2]
            Y7[real_rates τ2]
            Y8[inversion_depth τ2]
            Y9[slope_2s10s τ0]
            Y10[breakeven_inflation τ0]
        end
        subgraph CreditConditions
            CC1[ig_spread τ0]
            CC2[hy_spread τ0]
            CC3[cds_indices τ0]
            CC4[leveraged_loan_spread τ2]
            CC5[bank_lending_standards τ5]
            CC6[credit_impulse τ4]
            CC7[private_credit_growth τ5]
            CC8[distress_ratio τ4]
            CC9[default_rate τ4]
            CC10[covenant_lite_share τ5]
        end
        subgraph FXState
            FX1[dxy τ0]
            FX2[eurusd τ0]
            FX3[usdjpy τ0]
            FX4[usdcny τ0]
            FX5[em_fx_index τ0]
            FX6[fx_vol_surface τ0]
            FX7[carry_trade_profitability τ2]
            FX8[reserve_currency_shares τ5]
            FX9[sdr_composition τ6]
        end
        subgraph LiquidityState
            L1[fed_reverse_repo τ2]
            L2[treasury_general_account τ2]
            L3[bank_reserves τ3]
            L4[money_market_stress τ0]
            L5[repo_rate_spread τ0]
            L6[collateral_scarcity τ2]
            L7[global_m2_growth τ4]
            L8[dollar_funding_stress τ0]
            L9[cross_currency_basis τ0]
        end
        subgraph EquityMarketState
            EQ1[broad_indices τ0]
            EQ2[vix τ0]
            EQ3[vol_term_structure τ0]
            EQ4[sector_rotation τ2]
            EQ5[breadth τ2]
            EQ6[earnings_revision_ratio τ3]
            EQ7[buyback_pace τ5]
            EQ8[ipo_issuance τ4]
            EQ9[margin_debt τ4]
            EQ10[put_call_ratio τ0]
        end
        subgraph CryptoState
            CR1[btc τ0]
            CR2[eth τ0]
            CR3[stablecoin_supply τ2]
            CR4[defi_tvl τ2]
            CR5[crypto_vol τ0]
            CR6[institutional_flows τ3]
        end
    end
```

**Why this decomposition?** Markets are the world's real-time information system. The yield curve integrates growth expectations, inflation forecasts, and risk premia into a single object. Credit conditions propagate through the real economy with a lag. FX reflects relative monetary policy and capital flows. Liquidity is the plumbing — when it breaks (repo crisis, dollar squeeze), everything else breaks. Equities and crypto are the most reflexive, sentiment-driven markets. Each sub-system has its own frequency: yield levels are tick-level, but credit impulse is monthly.

---

## 4. Macroeconomic Layer

**Frequency**: τ3–τ5 (weekly to quarterly) · **Fields**: 67 (per country)

The real economy: GDP, inflation, labor, fiscal position, trade, and housing. This is instantiated per country — each `Country` entity contains a full `MacroEconomy`.

```mermaid
graph TD
    subgraph MacroEconomy
        subgraph OutputAndGrowth
            OG1["gdp_nowcast τ2 (w=2.0)"]
            OG2[gdp_official<br/>ObservedSlow]
            OG3[industrial_production τ4]
            OG4[capacity_utilization τ4]
            OG5["pmi_manufacturing τ4 (w=1.5)"]
            OG6["pmi_services τ4 (w=1.5)"]
            OG7[retail_sales τ4]
            OG8[new_orders τ4]
            OG9[potential_growth τ6]
        end
        subgraph InflationState
            IN1["headline_cpi τ4 (w=2.0)"]
            IN2["core_cpi τ4 (w=2.5)"]
            IN3[pce_deflator τ4]
            IN4[ppi τ4]
            IN5[wage_growth τ4]
            IN6[rent_inflation τ4]
            IN7[expectations_1y τ4]
            IN8["expectations_5y τ4 (w=2.0)"]
            IN9[sticky_vs_flexible τ4]
            IN10[supply_driven_share τ4]
        end
        subgraph LaborMarket
            LM1[unemployment_rate τ4]
            LM2["nfp_change τ4 (w=1.5)"]
            LM3[initial_claims τ3]
            LM4[continuing_claims τ3]
            LM5[job_openings τ4]
            LM6[quits_rate τ4]
            LM7[lfpr τ4]
            LM8[avg_hourly_earnings τ4]
            LM9[unit_labor_costs τ5]
            LM10[immigration_flow τ6]
        end
        subgraph FiscalState
            FS1[debt_to_gdp τ5]
            FS2[deficit_to_gdp τ5]
            FS3["interest_expense_share τ5 (w=2.0)"]
            FS4[fiscal_impulse τ5]
            FS5[spending_composition τ6]
            FS6[tax_revenue_trend τ5]
            FS7[debt_maturity_profile τ5]
            FS8[sovereign_cds τ0]
            FS9["debt_ceiling_proximity τ3 (w=3.0)"]
        end
        subgraph TradeBalance
            TB1[current_account τ5]
            TB2[trade_balance τ4]
            TB3[capital_flows_net τ4]
            TB4[fdi_flows τ5]
            TB5[terms_of_trade τ4]
            TB6[tariff_effective_rate τ4]
            TB7[sanctions_exposure τ3]
        end
        subgraph HousingMarket
            HM1[home_price_index τ4]
            HM2[housing_starts τ4]
            HM3[mortgage_rate τ2]
            HM4[existing_home_sales τ4]
            HM5[affordability τ4]
            HM6[delinquency_rate τ5]
        end
    end
```

**Why this decomposition?** GDP is a lagging composite, so we track high-frequency nowcasts alongside official releases (with revision risk). Inflation is decomposed into components because the policy response depends on *which* prices are rising — rent vs food vs energy. Labor markets lead the cycle (claims are weekly). Fiscal position matters increasingly in a post-2020 world of large deficits. Trade and housing are transmission channels for monetary policy. Every field has a natural publication cadence built into its period.

---

## 5. Political Layer

**Frequency**: τ4–τ7 (monthly to multi-year) · **Fields**: 42 (per country)

Governance structures: executive power, legislative capacity, judicial independence, geopolitical dynamics, and institutional quality. These determine the rules of the game.

```mermaid
graph TD
    subgraph PoliticalLayer
        subgraph ExecutiveState
            EX1[approval_rating τ3]
            EX2[political_capital τ4]
            EX3[executive_coherence τ4]
            EX4["election_proximity τ3 (w=2.0)"]
            EX5[lame_duck_index τ4]
            EX6[cabinet_stability τ4]
        end
        subgraph LegislativeState
            LE1[gridlock τ4]
            LE2[majority_margin τ5]
            LE3[bipartisan_capacity τ5]
            LE4[pending_legislation_risk τ3]
            LE5[regulatory_pipeline τ4]
        end
        subgraph JudicialState
            JU1[independence τ6]
            JU2[pending_landmark_cases τ4]
            JU3["regulatory_uncertainty τ4 (w=1.5)"]
        end
        subgraph GeopoliticalState
            GP1["conflict_risk τ2 (w=3.0)"]
            GP2[alliance_cohesion τ5]
            GP3["great_power_tension τ3 (w=2.0)"]
            GP4["nuclear_risk τ4 (w=5.0)"]
            GP5[sanctions_regime τ3]
            GP6[arms_trade τ5]
            GP7[territorial_disputes τ5]
            GP8[cyber_conflict τ3]
            GP9[space_competition τ5]
            GP10[economic_coercion τ3]
        end
        subgraph InstitutionalQuality
            IQ1[rule_of_law τ6]
            IQ2[corruption τ6]
            IQ3[state_capacity τ6]
            IQ4[property_rights τ6]
            IQ5[press_freedom τ6]
            IQ6["democratic_backsliding τ6 (w=2.0)"]
            IQ7[social_trust τ6]
        end
    end
```

**Why this decomposition?** Markets react to political events (elections, policy announcements) but the *structural* political variables — institutional quality, rule of law — are the slow-moving foundations. Geopolitical state gets high loss weights because conflicts have outsized impact on all other layers. Nuclear risk at w=5.0 is the most heavily weighted political field: low probability, civilization-scale consequence.

---

## 6. Narrative & Belief Layer

**Frequency**: τ0–τ4 (sub-minute to monthly) · **Fields**: 35

Reflexivity in the world model. Media narratives, elite consensus, public sentiment, and investor positioning — beliefs that change reality by changing behavior.

```mermaid
graph TD
    subgraph NarrativeBeliefLayer
        subgraph MediaNarratives
            MN1[crisis_framing τ1]
            MN2[econ_doom_vs_boom τ2]
            MN3[geopolitical_fear τ2]
            MN4[tech_optimism τ3]
            MN5[inequality_salience τ3]
            MN6[climate_urgency τ4]
            MN7[media_fragmentation τ6]
            MN8[info_ecosystem_health τ6]
        end
        subgraph EliteConsensus
            EC1[davos_consensus τ5]
            EC2[cb_hawkishness τ4]
            EC3[ceo_confidence τ4]
            EC4[vc_risk_appetite τ4]
            EC5[techno_optimism τ4]
            EC6[deglobalization_belief τ5]
            EC7[ai_xrisk_belief τ5]
        end
        subgraph PublicSentiment
            PS1[consumer_confidence τ4]
            PS2[economic_anxiety τ3]
            PS3[institutional_trust τ5]
            PS4[polarization τ5]
            PS5["social_unrest_risk τ3 (w=2.0)"]
            PS6[migration_pressure τ4]
            PS7[birth_rate_sentiment τ6]
            PS8[techno_anxiety τ4]
        end
        subgraph InvestorPositioning
            IP1[equity_fund_flows τ3]
            IP2[bond_fund_flows τ3]
            IP3[mm_fund_flows τ3]
            IP4[hedge_fund_leverage τ4]
            IP5[cta_signal τ2]
            IP6[retail_sentiment τ2]
            IP7[institutional_rebalancing τ5]
            IP8["crowdedness_risk τ3 (w=1.5)"]
            IP9[short_interest τ3]
        end
    end
```

**Why this decomposition?** Soros's reflexivity: market participants' beliefs change the fundamentals. Media narratives frame the interpretation of data releases (the same jobs report reads differently under "doom" vs "boom" framing). Elite consensus (Davos, central bankers, VCs) sets investment and policy direction. Public sentiment drives consumption and political outcomes. Investor positioning is the mechanical bridge — fund flows *are* prices, and crowded trades create fragility.

---

## 7. Technology Layer

**Frequency**: τ5–τ7 (quarterly to multi-year) · **Fields**: 13

Long-run structural drivers. AI capability, biotech, quantum computing, and productivity — the forces that reshape the production function.

```mermaid
graph TD
    subgraph TechnologyLayer
        T1["ai_capability_frontier τ4 (w=3.0)"]
        T2[ai_adoption τ5]
        T3[ai_safety_governance τ5]
        T4[ai_compute_scaling τ4]
        T5[biotech_frontier τ5]
        T6[quantum_progress τ6]
        T7[robotics_deployment τ5]
        T8[fusion_progress τ6]
        T9[space_commercialization τ5]
        T10["productivity_growth τ6 (w=2.0)"]
        T11[automation_displacement τ5]
        T12[patent_activity τ5]
        T13[global_r_and_d τ6]
    end
```

**Why this decomposition?** Technology operates on longer timescales than markets but occasionally creates discontinuities (GPT-4, mRNA vaccines). AI gets the most fields and highest weights because it's the meta-technology — it accelerates every other field. Productivity growth is the single most important long-run economic variable. The layer is deliberately sparse because technology is hard to forecast; the model should learn what it can and be honest about epistemic limits.

---

## 8. Biological Layer

**Frequency**: τ3–τ6 (weekly to annual) · **Fields**: 16

Ecological systems: biodiversity, disease dynamics, agricultural biology. The living substrate that human systems depend on.

```mermaid
graph TD
    subgraph BiologicalLayer
        subgraph EcosystemState
            EC1["biodiversity_index τ6 (w=2.0)"]
            EC2[deforestation_rate τ5]
            EC3[ocean_acidification τ6]
            EC4[coral_reef_health τ5]
            EC5[fish_stock_status τ5]
            EC6["pollinator_decline τ6 (w=1.5)"]
        end
        subgraph DiseaseState
            DS1["pandemic_readiness τ4 (w=2.0)"]
            DS2["novel_pathogen_risk τ3 (w=2.0)"]
            DS3[vaccine_pipeline τ5]
            DS4["antimicrobial_resistance τ6 (w=1.5)"]
            DS5["zoonotic_spillover_risk τ5 (w=1.5)"]
        end
        subgraph AgriculturalBiology
            AB1[crop_disease_pressure τ4]
            AB2[soil_health_index τ6]
            AB3[seed_technology τ5]
            AB4[livestock_health τ4]
            AB5[aquaculture_output τ5]
        end
    end
```

**Why this decomposition?** COVID demonstrated that biological systems can suddenly dominate all other layers. Ecosystem collapse (biodiversity, pollinators) is a slow-burn risk with catastrophic tail outcomes. Disease state tracks pandemic readiness and emerging threats. Agricultural biology feeds into food system prices and food security. Each sub-system has elevated loss weights because biological risks are systematically underpriced by markets.

---

## 9. Infrastructure Layer

**Frequency**: τ1–τ6 (hourly to annual) · **Fields**: 27

Power grids, transport networks, telecoms, urban systems. The physical substrate of the economy — usually invisible until it breaks.

```mermaid
graph TD
    subgraph InfrastructureLayer
        subgraph PowerGrid
            PG1[generation_capacity τ4]
            PG2["grid_reliability τ2 (w=1.5)"]
            PG3[renewable_penetration τ5]
            PG4[storage_capacity τ5]
            PG5[peak_demand τ1]
            PG6[transmission_congestion τ2]
            PG7["blackout_risk τ3 (w=2.0)"]
        end
        subgraph TransportNetwork
            TN1[road_congestion τ1]
            TN2[rail_utilization τ2]
            TN3[aviation_load τ2]
            TN4[port_throughput τ3]
            TN5[last_mile_cost τ4]
            TN6[ev_adoption τ5]
            TN7[autonomous_vehicle_deployment τ6]
        end
        subgraph TelecomNetwork
            TC1[bandwidth_demand τ1]
            TC2[fiber_coverage τ5]
            TC3[satellite_constellation_capacity τ5]
            TC4[spectrum_allocation τ6]
            TC5[latency_critical_services τ2]
            TC6[fiveg_coverage τ5]
        end
        subgraph UrbanSystems
            US1[housing_inventory τ4]
            US2[construction_permits τ4]
            US3[commercial_vacancy τ4]
            US4[urban_density_trend τ6]
            US5[smart_city_index τ6]
            US6[public_transit_utilization τ2]
        end
    end
```

**Why this decomposition?** Infrastructure has the widest frequency spread: peak demand is hourly, but generation capacity changes monthly and renewable penetration is quarterly. Transport and telecom are separate because their failure modes are different. Urban systems connect to housing markets and demographics. Blackout risk gets elevated loss weight because grid failures cascade into every other system.

---

## 10. Cyber Layer

**Frequency**: τ2–τ5 (daily to quarterly) · **Fields**: 11

Cybersecurity threats and the digital ecosystem. A growing attack surface that increasingly affects physical infrastructure, financial systems, and geopolitics.

```mermaid
graph TD
    subgraph CyberLayer
        subgraph CyberThreatLandscape
            CT1[attack_surface τ3]
            CT2["ransomware_frequency τ3 (w=1.5)"]
            CT3["nation_state_activity τ4 (w=2.0)"]
            CT4["zero_day_inventory τ3 (w=2.0)"]
            CT5["critical_infrastructure_targeting τ2 (w=2.0)"]
            CT6["supply_chain_compromise_risk τ4 (w=1.5)"]
        end
        subgraph DigitalEcosystem
            DE1[platform_concentration τ5]
            DE2[data_sovereignty_regulation τ5]
            DE3[ai_generated_content_share τ4]
            DE4[digital_identity_adoption τ5]
            DE5[open_source_health τ5]
        end
    end
```

**Why this decomposition?** Cyber threats are one of the few domains where the attack surface grows faster than defenses. Nation-state activity and zero-day inventory track the offensive capability landscape. Critical infrastructure targeting bridges cyber to physical systems. The digital ecosystem fields track structural features of the internet itself — platform concentration, data sovereignty — that shape what kinds of attacks are possible.

---

## 11. Space Layer

**Frequency**: τ3–τ6 (weekly to annual) · **Fields**: 9

The orbital environment and space economy. Increasingly relevant as satellite internet, space tourism, and orbital congestion become economic factors.

```mermaid
graph TD
    subgraph SpaceLayer
        subgraph OrbitalEnvironment
            OE1[active_satellites τ4]
            OE2[debris_density τ5]
            OE3["collision_risk τ3 (w=2.0)"]
            OE4[orbit_congestion_leo τ4]
        end
        subgraph SpaceEconomy
            SE1[launch_cost τ5]
            SE2[commercial_space_revenue τ5]
            SE3[space_tourism τ6]
            SE4[satellite_internet_subscribers τ5]
            SE5[space_mining_progress τ6]
        end
    end
```

**Why this decomposition?** Kessler syndrome (collision cascading) is a low-probability, high-consequence risk to communications, GPS, and military systems. Launch cost tracks the SpaceX-driven deflation curve. Satellite internet is already affecting telecom markets. The layer is small but connects to infrastructure, cyber, and geopolitics.

---

## 12. Health Layer

**Frequency**: τ3–τ6 (weekly to annual) · **Fields**: 10

Healthcare capacity and public health outcomes. Connects to demographics, labor markets, and fiscal spending.

```mermaid
graph TD
    subgraph HealthLayer
        subgraph HealthcareCapacity
            HC1[hospital_beds_per_capita τ5]
            HC2["icu_utilization τ3 (w=1.5)"]
            HC3[healthcare_worker_shortage τ4]
            HC4[telehealth_adoption τ5]
            HC5[pharma_pipeline τ5]
        end
        subgraph PublicHealth
            PH1["life_expectancy_trend τ6 (w=2.0)"]
            PH2[obesity_rate τ6]
            PH3["mental_health_crisis τ4 (w=1.5)"]
            PH4[substance_abuse_trend τ5]
            PH5["health_inequality τ6 (w=1.5)"]
        end
    end
```

**Why this decomposition?** COVID revealed that healthcare capacity is a binding constraint on economic activity. ICU utilization is the canary — when it spikes, policy responses follow. Mental health crisis and substance abuse are slow-burn risks that show up in labor force participation and productivity. Health inequality connects to political polarization and social trust.

---

## 13. Education Layer

**Frequency**: τ4–τ6 (monthly to annual) · **Fields**: 11

Education systems and workforce development. The pipeline that turns demographics into human capital.

```mermaid
graph TD
    subgraph EducationLayer
        subgraph EducationSystem
            ES1[enrollment_rate τ6]
            ES2[stem_graduation τ6]
            ES3["skill_gap_index τ5 (w=1.5)"]
            ES4[online_learning_penetration τ5]
            ES5[research_output τ5]
            ES6[university_funding τ6]
        end
        subgraph WorkforceDevelopment
            WD1[retraining_programs τ5]
            WD2[apprenticeship_rate τ6]
            WD3[remote_work_share τ4]
            WD4[gig_economy_share τ5]
            WD5[labor_mobility τ5]
        end
    end
```

**Why this decomposition?** Skill gaps are the binding constraint on technology adoption. STEM graduation rates constrain AI development. Remote work share reshapes commercial real estate and urban systems. These fields operate on long timescales but have compounding effects.

---

## 14. Demographics Layer

**Frequency**: τ7 (decadal) · **Fields**: 10

The slowest structural force: population, dependency ratios, urbanization, fertility. These are the tectonic plates of economics.

```mermaid
graph TD
    subgraph DemographicLayer
        D1[population_growth τ7]
        D2[dependency_ratio τ7]
        D3[urbanization τ7]
        D4[median_age τ7]
        D5[fertility_rate τ7]
        D6[life_expectancy τ7]
        D7[net_migration τ6]
        D8[education_attainment τ7]
        D9[human_capital_index τ7]
        D10["working_age_growth τ7 (w=2.0)"]
    end
```

**Why this decomposition?** Demographics is destiny — but slowly. Working-age population growth determines potential GDP growth decades in advance. Dependency ratios drive fiscal pressure. Migration is the one fast-moving demographic variable (annual vs decadal). The layer is per-country, embedded inside `Country`.

---

## 15. Legal & Regulatory Layer

**Frequency**: τ5–τ6 (quarterly to annual) · **Fields**: 11

The regulatory environment and rule of law. These fields determine the cost of doing business and the reliability of contracts.

```mermaid
graph TD
    subgraph LegalLayer
        subgraph RegulatoryEnvironment
            RE1[regulatory_burden_index τ5]
            RE2[antitrust_enforcement τ5]
            RE3[ip_protection_strength τ6]
            RE4["environmental_regulation τ5 (w=1.5)"]
            RE5[financial_regulation_stringency τ5]
            RE6[data_privacy_regulation τ5]
        end
        subgraph LegalSystem
            LS1["judicial_independence τ6 (w=2.0)"]
            LS2[contract_enforcement τ6]
            LS3["corruption_index τ6 (w=2.0)"]
            LS4["rule_of_law_index τ6 (w=2.0)"]
            LS5[international_arbitration τ5]
        end
    end
```

**Why this decomposition?** Regulatory environment changes at quarterly cadence (new rules, enforcement actions) while the underlying legal system quality is annual/structural. High loss weights on rule of law and corruption because these are among the strongest predictors of long-run economic outcomes.

---

## 16. Sector Layer

**Frequency**: τ3–τ5 (weekly to quarterly) · **Fields**: 19 (per sector)

Per-GICS sector dynamics: demand, supply, profitability, and structural forces. Instantiated dynamically for requested sectors.

```mermaid
graph TD
    subgraph Sector
        subgraph Demand
            SD1[demand_growth τ4]
            SD2[pricing_power τ4]
            SD3[order_backlog τ4]
            SD4[end_market_health τ4]
        end
        subgraph Supply
            SS1[capacity_utilization τ4]
            SS2[supply_chain_stress τ3]
            SS3[inventory_to_sales τ4]
            SS4[labor_availability τ4]
        end
        subgraph Profitability
            SP1[margins τ5]
            SP2[input_cost_pressure τ4]
            SP3[revenue_growth τ5]
            SP4[capex_cycle τ5]
        end
        subgraph Structural
            ST1[innovation_rate τ5]
            ST2["regulatory_risk τ4 (w=1.5)"]
            ST3["disruption_risk τ5 (w=2.0)"]
            ST4[esg_pressure τ5]
            ST5[m_and_a_activity τ5]
            ST6[concentration τ6]
        end
        DQ[data_quality τ5]
    end
```

**Why this decomposition?** Sectors are the natural unit of equity analysis. Demand/supply/profitability captures the operating leverage cycle. Structural fields (disruption risk, concentration) drive long-run sector returns. Each sector is a dynamic entity — add `entities={"sector_tech": Sector(), "sector_energy": Sector()}` to include sector-level analysis in a projection.

---

## 17. Supply Chain Layer

**Frequency**: τ2–τ4 (daily to monthly) · **Fields**: 9 (per node)

Graph structure: each supply chain node has concentration, inventory, lead time, and fragility metrics. These form a network that propagates shocks.

```mermaid
graph TD
    subgraph SupplyChainNode
        SC1[upstream_concentration τ4]
        SC2[downstream_concentration τ4]
        SC3[inventory τ3]
        SC4[lead_time τ3]
        SC5[logistics_friction τ2]
        SC6["bottleneck_severity τ2 (w=2.0)"]
        SC7[substitutability τ5]
        SC8[geographic_risk τ4]
        SC9["single_point_of_failure τ5 (w=3.0)"]
    end
```

**Why this decomposition?** Supply chain disruptions (COVID, Suez Canal, semiconductor shortages) propagate non-linearly. The key insight is that *concentration* and *substitutability* determine fragility. Single point of failure gets the highest weight (w=3.0) because it identifies the nodes where disruption is catastrophic. The layer is embedded inside `Business` entities, creating a firm-level supply chain graph.

---

## 18. Business Layer

**Frequency**: τ2–τ5 (daily to quarterly) · **Fields**: 57 (per firm)

Full firm decomposition: financials, operations, strategy, market position, and risk. The richest dynamic entity in the schema.

```mermaid
graph TD
    subgraph Business
        subgraph FirmFinancials
            FF1[revenue τ5]
            FF2["revenue_growth τ5 (w=2.0)"]
            FF3[cogs τ5]
            FF4["gross_margin τ5 (w=2.0)"]
            FF5[opex τ5]
            FF6["operating_margin τ5 (w=2.0)"]
            FF7[net_income τ5]
            FF8["fcf τ5 (w=2.5)"]
            FF9[cash τ5]
            FF10[total_debt τ5]
            FF11["net_debt_to_ebitda τ5 (w=2.0)"]
            FF12["interest_coverage τ5 (w=2.0)"]
            FF13["covenant_headroom τ5 (w=3.0)"]
            FF14["maturity_wall τ5 (w=2.5)"]
            FF15[working_capital τ5]
            FF16[capex τ5]
            FF17[share_count τ5]
            FF18[insider_transactions τ4]
        end
        subgraph FirmOperations
            FO1[capacity τ4]
            FO2[utilization τ4]
            FO3[backlog τ4]
            FO4["pricing_power τ4 (w=1.5)"]
            FO5[customer_concentration τ5]
            FO6[supplier_concentration τ5]
            FO7[quality_incidents τ4]
            FO8[headcount τ5]
            FO9[employee_satisfaction τ5]
            FO10[tech_debt τ5]
        end
        subgraph FirmStrategy
            FS1[roadmap_clarity τ5]
            FS2[capex_plan τ5]
            FS3[m_and_a_appetite τ5]
            FS4[geographic_expansion τ5]
            FS5[product_pipeline τ5]
            FS6["moat_durability τ6 (w=2.0)"]
            FS7["management_quality τ5 (w=2.0)"]
            FS8["capital_allocation τ5 (w=2.0)"]
            FS9[governance_quality τ5]
            FS10[esg_posture τ5]
        end
        subgraph FirmMarketPosition
            FM1[equity_price τ0]
            FM2[implied_vol τ0]
            FM3[credit_spread τ0]
            FM4[analyst_consensus τ3]
            FM5[short_interest τ3]
            FM6[institutional_ownership τ5]
            FM7[pe_ratio τ2]
            FM8[ev_ebitda τ2]
        end
        subgraph FirmRisk
            FR1[regulatory_exposure τ4]
            FR2[litigation_risk τ4]
            FR3[cyber_vulnerability τ4]
            FR4["key_person_risk τ5 (w=2.0)"]
            FR5[supply_chain_fragility τ4]
            FR6["geopolitical_exposure τ4 (w=1.5)"]
            FR7[climate_transition_risk τ6]
            FR8["tech_obsolescence τ5 (w=2.0)"]
        end
        subgraph LatentAndOutputs
            LO1["latent_health (w=3.0)"]
            LO2["latent_momentum (w=2.0)"]
            LO3["latent_tail_risk (w=4.0)"]
            LO4["recommended_action (w=3.0)"]
            LO5["fair_value_estimate (w=2.0)"]
        end
    end
```

**Why this decomposition?** Business is where the macro-financial-political worlds meet the micro-reality of a single firm. The 57-field decomposition covers the full analyst toolkit: financial statements (income, balance sheet, cash flow), operational metrics (capacity, backlog, quality), strategic positioning (moat, management, M&A), market pricing (equity, credit, vol), and risk factors. Latent variables (health, momentum, tail_risk) are unobserved — the model learns to infer them from the observed fields. Each firm also embeds a `SupplyChainNode`, connecting the firm-level graph to the supply chain network.

---

## 19. Individual Layer

**Frequency**: τ2–τ5 (daily to quarterly) · **Fields**: 27 (per person)

Psychological decomposition of decision-makers: cognition, incentives, network position, and current state. The most speculative layer, designed for modeling CEOs, central bankers, and political leaders.

```mermaid
graph TD
    subgraph Individual
        subgraph PersonCognitive
            PC1[decision_style τ5]
            PC2[risk_appetite τ4]
            PC3[time_horizon τ5]
            PC4[belief_update_speed τ4]
            PC5[cognitive_load τ3]
            PC6[ideological_priors τ6]
        end
        subgraph PersonIncentives
            PI1[compensation_structure τ5]
            PI2[career_incentives τ5]
            PI3[reputation_concerns τ4]
            PI4[legal_exposure τ4]
            PI5[legacy_concerns τ5]
            PI6[peer_pressure τ4]
        end
        subgraph PersonNetwork
            PN1[formal_authority τ5]
            PN2[network_centrality τ5]
            PN3[trusted_advisors τ5]
            PN4[board_relationships τ5]
            PN5[political_connections τ5]
            PN6[media_influence τ4]
        end
        subgraph PersonState
            PS1[stress τ3]
            PS2[health_energy τ4]
            PS3[confidence τ3]
            PS4[current_focus τ3]
            PS5[public_statements_tone τ2]
            PS6["private_info_proxy τ3 (w=3.0)"]
        end
        subgraph OutputHeads
            OH1["projected_actions (w=3.0)"]
            OH2["action_timing (w=2.0)"]
            OH3["surprise_risk (w=4.0)"]
        end
    end
```

**Why this decomposition?** Individual decision-makers can move markets. A Fed chair's hawkishness, a CEO's risk appetite, a president's belligerence — these matter. The decomposition follows behavioral economics: cognitive style determines *how* information is processed; incentives determine *what* actions are likely; network position determines *influence*; current state determines *timing*. Private information proxy (w=3.0) is the holy grail — what does this person know that the market doesn't?

---

## Meta Layers

### Event Tape

**Frequency**: τ0–τ1 (real-time) · **Fields**: 10

Dense-in-time, compressed-in-space stream of world events.

```mermaid
graph TD
    subgraph EventTape
        EV1["news_embedding τ0 (4×8)"]
        EV2["social_signal τ0 (2×4)"]
        EV3["filing_events τ2 (2×4)"]
        EV4["earnings_call_signal τ5 (2×4)"]
        EV5["policy_announcement τ0 (w=2.0)"]
        EV6["conflict_event τ0 (w=3.0)"]
        EV7["disaster_event τ0 (w=2.0)"]
        EV8[trade_data_release τ4]
        EV9["central_bank_comms τ4 (w=2.0)"]
        EV10["election_event τ3 (w=2.0)"]
    end
```

### Data Channel Trust

**Frequency**: varies · **Fields**: 17

Meta-epistemic calibration: how much should the model trust each data source?

```mermaid
graph TD
    subgraph DataChannelTrust
        subgraph GovernmentStats
            GT1[trust_bls τ5]
            GT2[trust_census τ6]
            GT3[trust_fed τ5]
            GT4[trust_foreign_gov τ5]
        end
        subgraph MarketData
            MD1[trust_exchange τ4]
            MD2[trust_otc τ4]
            MD3[trust_credit_rating τ5]
        end
        subgraph AlternativeData
            AD1[trust_satellite τ5]
            AD2[trust_web_scraping τ4]
            AD3[trust_social_sentiment τ3]
            AD4[trust_survey τ4]
        end
        subgraph Corporate
            CD1[trust_gaap τ5]
            CD2["trust_mgmt_guidance τ5 (w=2.0)"]
            CD3["trust_non_gaap τ5 (w=1.5)"]
        end
        subgraph Meta
            ME1["overall_epistemic_state (w=2.0)"]
            ME2[information_advantage]
            ME3["adversarial_info_risk τ3 (w=2.0)"]
        end
    end
```

### Regime State

**Frequency**: τ5–τ7 (quarterly to decadal) · **Fields**: 17

The compressed world state. Regime variables determine which causal channels are active — in a recession regime, different dynamics dominate than in expansion.

```mermaid
graph TD
    subgraph RegimeState
        subgraph EconomicRegime
            ER1["growth_regime τ5 (w=5.0)"]
            ER2["inflation_regime τ5 (w=5.0)"]
            ER3["financial_cycle τ5 (w=4.0)"]
            ER4["credit_cycle τ5 (w=4.0)"]
            ER5["liquidity_regime τ4 (w=4.0)"]
        end
        subgraph GeopoliticalRegime
            GR1["cooperation_vs_fragmentation τ6 (w=4.0)"]
            GR2["peace_vs_conflict τ5 (w=5.0)"]
            GR3["hegemonic_stability τ7 (w=3.0)"]
            GR4["globalization_vs_autarky τ6 (w=3.0)"]
        end
        subgraph TechnologyRegime
            TR1["ai_acceleration τ5 (w=4.0)"]
            TR2["energy_transition_phase τ6 (w=3.0)"]
            TR3["productivity_regime τ6 (w=3.0)"]
        end
        subgraph SystemicRisk
            SR1["fragility τ4 (w=5.0)"]
            SR2["reflexivity_intensity τ4 (w=4.0)"]
            SR3["tail_risk_concentration τ4 (w=5.0)"]
            SR4["black_swan_proximity τ3 (w=5.0)"]
        end
        CS["compressed_world_state (4×8, w=3.0)"]
    end
```

### Intervention Space

**Frequency**: varies · **Fields**: 13

What-if analysis: declare a policy intervention, predict the counterfactual effects.

```mermaid
graph TD
    subgraph InterventionSpace
        subgraph PolicyInterventions
            PI1[monetary_policy_change τ4]
            PI2[fiscal_policy_change τ5]
            PI3[regulatory_action τ4]
            PI4[sanctions_change τ3]
            PI5[trade_policy_change τ4]
            PI6["military_action τ3 (w=5.0)"]
            PI7[firm_strategic_action τ5]
            PI8[technology_release τ4]
            PI9[market_intervention τ3]
        end
        subgraph CounterfactualHeads
            CF1["effect_3m (w=3.0)"]
            CF2["effect_12m (w=2.0)"]
            CF3["second_order_effects (w=2.0)"]
            CF4["unintended_consequences (w=3.0)"]
        end
    end
```

### Forecast Bundle

**Frequency**: output heads · **Fields**: 32

Structured output predictions: recession probability, credit stress, conflict escalation, and more.

```mermaid
graph TD
    subgraph ForecastBundle
        subgraph MacroForecast
            MF1["recession_prob_3m (w=5.0)"]
            MF2["recession_prob_12m (w=4.0)"]
            MF3["gdp_growth_3m (w=3.0)"]
            MF4["gdp_growth_12m (w=2.0)"]
            MF5["inflation_path_12m (w=3.0)"]
            MF6["rates_path_12m (w=3.0)"]
            MF7["unemployment_path_12m (w=2.0)"]
        end
        subgraph FinancialForecast
            FF1["credit_stress_3m (w=4.0)"]
            FF2["equity_regime_3m (w=3.0)"]
            FF3["vol_regime_3m (w=3.0)"]
            FF4["sector_rotation_3m (w=2.0)"]
            FF5["curve_shape_3m (w=2.0)"]
            FF6["fx_regime_3m (w=2.0)"]
            FF7["liquidity_crisis_prob (w=5.0)"]
        end
        subgraph GeopoliticalForecast
            GF1["conflict_escalation_3m (w=5.0)"]
            GF2["sanctions_change_3m (w=3.0)"]
            GF3["alliance_shift_12m (w=2.0)"]
            GF4["regime_change_prob (w=4.0)"]
            GF5["election_outcome (w=3.0)"]
        end
        subgraph BusinessForecast
            BF1["revenue_surprise (w=3.0)"]
            BF2["margin_trajectory (w=3.0)"]
            BF3["default_prob_12m (w=5.0)"]
            BF4["strategic_pivot_prob (w=3.0)"]
            BF5["m_and_a_prob (w=2.0)"]
            BF6["mgmt_change_prob (w=3.0)"]
        end
        subgraph UncertaintyDecomposition
            UD1["aleatoric_macro (w=2.0)"]
            UD2["aleatoric_geopolitical (w=2.0)"]
            UD3["epistemic_data_gaps (w=2.0)"]
            UD4[epistemic_model_limits]
            UD5["scenario_divergence (w=3.0)"]
            UD6["calibration_score (w=3.0)"]
        end
    end
```

---

## Summary statistics

| Layer | Fields | Frequency range | Dynamic? |
|-------|--------|----------------|----------|
| Physical | 17 | τ0–τ7 | No |
| Resources | 45 | τ0–τ6 | No |
| Financial | 68 | τ0–τ5 | No |
| Macro | 67 | τ0–τ6 | Per country |
| Political | 42 | τ2–τ6 | Per country |
| Narratives | 35 | τ1–τ6 | No |
| Technology | 13 | τ4–τ6 | No |
| Biology | 16 | τ3–τ6 | No |
| Infrastructure | 27 | τ1–τ6 | No |
| Cyber | 11 | τ2–τ5 | No |
| Space | 9 | τ3–τ6 | No |
| Health | 10 | τ3–τ6 | No |
| Education | 11 | τ4–τ6 | No |
| Demographics | 10 | τ6–τ7 | Per country |
| Legal | 11 | τ5–τ6 | No |
| Sector | 19 | τ3–τ6 | Per sector |
| Supply Chain | 9 | τ2–τ5 | Per node |
| Business | 57 | τ0–τ6 | Per firm |
| Individual | 27 | τ2–τ6 | Per person |
| Events | 10 | τ0–τ5 | No |
| Trust | 17 | τ3–τ6 | No |
| Regime | 17 | τ3–τ7 | No |
| Interventions | 13 | τ3–τ5 | No |
| Forecasts | 32 | output | No |
| **Total** | **857** | **τ0–τ7** | |

**Temporal frequency classes:**

| Class | Period (ticks) | Real-world cadence | Description |
|-------|----------------|-------------------|-------------|
| τ0 | 1 | Sub-minute | Market prices, breaking news |
| τ1 | 4 | Hourly | Grid load, intraday commodities |
| τ2 | 16 | Daily | Commodity closes, port congestion |
| τ3 | 48 | Weekly | Jobless claims, inventories |
| τ4 | 192 | Monthly | CPI, PMI, housing starts |
| τ5 | 576 | Quarterly | GDP, earnings, capex |
| τ6 | 2304 | Annual | Demographics, infrastructure |
| τ7 | 4608 | Multi-year | Regime changes, tech diffusion |
