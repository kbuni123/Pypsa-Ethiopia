import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

try:
    import pypsa
    PYPSA_OK = True
except ImportError:
    PYPSA_OK = False

st.title("PyPSA-Ethiopia - Working Version")

if not PYPSA_OK:
    st.error("Install PyPSA first: pip install pypsa")
    st.stop()

if st.button("Run Ethiopia Energy Model"):
    
    # Create network
    n = pypsa.Network()
    
    # Set time - 24 hours only
    n.set_snapshots(pd.date_range('2023-01-01', periods=24, freq='H'))
    
    # Add buses
    n.add("Bus", "Addis_Ababa")
    n.add("Bus", "Bahir_Dar")
    
    # Add load - FIXED VALUES
    load_24h = [300, 280, 260, 250, 270, 320, 400, 450, 420, 380,
                360, 350, 340, 350, 360, 400, 450, 500, 480, 450, 420, 380, 350, 320]
    
    n.add("Load", "addis_load", bus="Addis_Ababa", p_set=load_24h)
    n.add("Load", "bahir_load", bus="Bahir_Dar", p_set=[x/2 for x in load_24h])
    
    # Add generators
    # Solar
    solar_24h = [0,0,0,0,0,0.1,0.3,0.5,0.7,0.8,0.9,1.0,1.0,0.9,0.8,0.7,0.5,0.3,0.1,0,0,0,0,0]
    
    n.add("Generator", "solar_addis",
          bus="Addis_Ababa",
          carrier="solar",
          p_nom_extendable=True,
          capital_cost=600,
          marginal_cost=0,
          p_max_pu=solar_24h)
    
    # Wind
    wind_24h = [0.3, 0.4, 0.6, 0.7, 0.5, 0.3, 0.2, 0.1, 0.2, 0.4,
                0.6, 0.8, 0.7, 0.6, 0.5, 0.4, 0.5, 0.7, 0.8, 0.6, 0.5, 0.4, 0.3, 0.3]
    
    n.add("Generator", "wind_bahir",
          bus="Bahir_Dar", 
          carrier="wind",
          p_nom_extendable=True,
          capital_cost=1200,
          marginal_cost=0,
          p_max_pu=wind_24h)
    
    # Gas backup
    n.add("Generator", "gas_backup",
          bus="Addis_Ababa",
          carrier="gas",
          p_nom_extendable=True,
          capital_cost=800,
          marginal_cost=50)
    
    # Transmission
    n.add("Line", "addis_bahir",
          bus0="Addis_Ababa", bus1="Bahir_Dar",
          s_nom=500, length=300, r=0.01, x=0.1)
    
    st.write("Network built successfully")
    st.write(f"Total demand: {n.loads_t.p_set.sum().sum():.0f} MWh")
    
    # Solve
    try:
        status, condition = n.optimize(solver_name='highs')
        
        st.write(f"Solver status: {status}")
        
        if status == "ok":
            # Results
            total_cost = n.objective
            total_capacity = n.generators.p_nom_opt.sum()
            total_generation = n.generators_t.p.sum().sum()
            
            st.success("Optimization completed!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Cost", f"${total_cost:,.0f}")
            with col2:
                st.metric("Total Generation", f"{total_generation:.0f} MWh")  
            with col3:
                renewable_gen = (n.generators_t.p[['solar_addis', 'wind_bahir']].sum().sum() 
                               if all(g in n.generators_t.p.columns for g in ['solar_addis', 'wind_bahir']) 
                               else 0)
                renewable_share = (renewable_gen / total_generation * 100) if total_generation > 0 else 0
                st.metric("Renewable Share", f"{renewable_share:.1f}%")
            
            # Show capacity breakdown
            cap_data = {}
            for carrier in ['solar', 'wind', 'gas']:
                gens = n.generators[n.generators.carrier == carrier]
                if len(gens) > 0:
                    cap_data[carrier] = gens.p_nom_opt.sum()
            
            if cap_data:
                fig = px.bar(x=list(cap_data.keys()), y=list(cap_data.values()),
                           title="Installed Capacity by Technology",
                           labels={'x': 'Technology', 'y': 'Capacity (MW)'})
                st.plotly_chart(fig)
            
            # Success message
            if total_cost > 0 and total_generation > 0:
                st.balloons()
                st.success("Model working correctly!")
            else:
                st.error("Still getting zero results")
        
        else:
            st.error(f"Optimization failed: {status} - {condition}")
            
    except Exception as e:
        st.error(f"Solving failed: {e}")
